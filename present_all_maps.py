import argparse
import os
import sys
from typing import Dict, Optional, Tuple, List

import geopandas as gpd
import pandas as pd


def log(msg: str):
    print(msg, flush=True)


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        if gdf.crs is None:
            return gdf.set_crs(epsg=4326)
        return gdf.to_crs(epsg=4326)
    except Exception:
        return gdf


def scan_dir(dir_path: str) -> Dict[str, Dict[str, str]]:
    layers: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(dir_path):
        return layers
    for fn in os.listdir(dir_path):
        p = os.path.join(dir_path, fn)
        if not os.path.isfile(p):
            continue
        base, ext = os.path.splitext(fn)
        ext = ext.lower()
        if ext in {".geojson", ".geoparquet", ".parquet", ".csv"}:
            layers.setdefault(base.lower(), {})[ext] = p
    return layers


def scan_layers(primary: str, secondary: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    layers = scan_dir(primary)
    if secondary and os.path.isdir(secondary):
        extra = scan_dir(secondary)
        for base, exts in extra.items():
            if base in layers:
                layers[base].update(exts)
            else:
                layers[base] = dict(exts)
    return layers


def resolve_by_keywords(layers: Dict[str, Dict[str, str]], keywords: List[str]) -> Optional[str]:
    for key, entry in layers.items():
        if any(kw in key for kw in keywords):
            for ext in (".geoparquet", ".parquet", ".geojson", ".csv"):
                if ext in entry:
                    return entry[ext]
    for entry in layers.values():
        for path in entry.values():
            name = os.path.basename(path).lower()
            if any(kw in name for kw in keywords):
                return path
    return None


def load_any(path: str) -> Optional[gpd.GeoDataFrame]:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".geoparquet", ".parquet"}:
            return gpd.read_parquet(path)
        if ext == ".geojson":
            return gpd.read_file(path)
        if ext == ".csv":
            df = pd.read_csv(path, low_memory=False)
            if {"lon", "lat"}.issubset(df.columns):
                return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
            if {"centroid_lon", "centroid_lat"}.issubset(df.columns):
                return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["centroid_lon"], df["centroid_lat"]), crs=4326)
            return gpd.GeoDataFrame(df, geometry=None)
        return gpd.read_file(path)
    except Exception as e:
        log(f"[WARN] Failed to read {path}: {e}")
        return None


def center_from_bounds(gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    try:
        g = ensure_wgs84(gdf)
        minx, miny, maxx, maxy = g.total_bounds
        return ((miny + maxy) / 2, (minx + maxx) / 2)
    except Exception:
        return (37.5407, -77.4360)  # Richmond approx


def map_service_lines(gdf: gpd.GeoDataFrame, out_html: str, max_points: int = 50000):
    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster
    except ImportError:
        log("[ERROR] folium is required. Install with: pip install folium")
        return

    g = ensure_wgs84(gdf)
    if len(g) == 0:
        log("[WARN] Service lines empty; skipping map.")
        return

    lat, lon = center_from_bounds(g)
    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")

    if len(g) > max_points:
        coords = list(zip(g.geometry.y, g.geometry.x))
        HeatMap(coords, radius=6, blur=8, max_zoom=13).add_to(m)
        folium.map.LayerControl().add_to(m)
        log(f"[OK] HeatMap with {len(coords)} points")
    else:
        mc = MarkerCluster().add_to(m)
        fields = [c for c in ["address", "bothsidesstatus", "neighborhood_name"] if c in g.columns]
        for _, row in g.iterrows():
            try:
                y, x = row.geometry.y, row.geometry.x
            except Exception:
                continue
            popup = None
            if fields:
                parts = [f"{f}: {row.get(f)}" for f in fields]
                popup = folium.Popup("<br/>".join(parts), max_width=300)
            folium.CircleMarker(location=[y, x], radius=3, color="#1f78b4", fill=True, fill_opacity=0.6, popup=popup).add_to(mc)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    log(f"[OK] Wrote map -> {out_html}")


def _make_colormap(series: pd.Series):
    from branca.colormap import LinearColormap
    s = pd.to_numeric(series, errors="coerce")
    vmin, vmax = float(s.quantile(0.05)), float(s.quantile(0.95))
    if vmin == vmax:
        vmin, vmax = float(s.min()), float(s.max())
    return LinearColormap(["#fee8c8", "#fdbb84", "#e34a33"], vmin=vmin, vmax=vmax)


def map_polygons(gdf: gpd.GeoDataFrame, out_html: str, name_field: Optional[str] = None, color_by: Optional[str] = None, boundaries_only: bool = False):
    try:
        import folium
    except ImportError:
        log("[ERROR] folium is required. Install with: pip install folium")
        return

    g = ensure_wgs84(gdf).dropna(subset=["geometry"])  # type: ignore[arg-type]
    if len(g) == 0:
        log("[WARN] Polygon layer empty; skipping map.")
        return

    lat, lon = center_from_bounds(g)
    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")

    tooltip = None
    if name_field and name_field in g.columns:
        tooltip = folium.features.GeoJsonTooltip(fields=[name_field], aliases=["Name"])  # type: ignore[arg-type]

    style_function = (lambda _: {"fill": False, "color": "#444", "weight": 1.0}) if boundaries_only else None

    if not boundaries_only and color_by and color_by in g.columns:
        cmap = _make_colormap(g[color_by])

        def style_fn(feat):
            val = feat["properties"].get(color_by)
            try:
                col = cmap(float(val)) if val is not None else "#cccccc"
            except Exception:
                col = "#cccccc"
            return {"fillColor": col, "color": "#666666", "weight": 0.8, "fillOpacity": 0.7}

        style_function = style_fn
        cmap.caption = color_by
        cmap.add_to(m)

    if style_function is None:
        def style_fn_simple(_):
            return {"fillColor": "#9ecae1", "color": "#6baed6", "weight": 0.8, "fillOpacity": 0.6}

        style_function = style_fn_simple

    folium.GeoJson(
        g.to_json(default=str),
        name=os.path.basename(out_html),
        style_function=style_function,
        tooltip=tooltip,
    ).add_to(m)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    log(f"[OK] Wrote map -> {out_html}")


def choropleth_enriched(gdf: gpd.GeoDataFrame, out_dir: str, name_field: Optional[str]):
    # Lead metrics
    for c in ["pct_known_lead", "pct_unknown", "pct_known_nonlead", "pct_known_lead_total", "pct_known_nonlead_total"]:
        if c in gdf.columns:
            map_polygons(gdf, os.path.join(out_dir, f"neighborhoods_enriched_{c}.html"), name_field=name_field, color_by=c)
    # Race metrics
    for c in ["pct_black", "pct_white", "pct_hispanic", "pct_asian", "pct_native", "pct_hipi"]:
        if c in gdf.columns:
            title = c.replace("pct_", "Pct ")
            map_polygons(gdf, os.path.join(out_dir, f"neighborhoods_enriched_{c}.html"), name_field=name_field, color_by=c)


def main():
    ap = argparse.ArgumentParser(description="Generate all interactive HTML maps (core layers + enriched neighborhoods).")
    ap.add_argument("--data-dir", default="data_raw", help="Folder with raw files")
    ap.add_argument("--cache-dir", default="rva_layers", help="Optional cache folder to search for polygon sources")
    ap.add_argument("--derived-dir", default="data_derived", help="Folder with derived outputs")
    ap.add_argument("--out-dir", default=os.path.join("outputs", "interactive"), help="Folder for HTML outputs")
    ap.add_argument("--race-col", default=None, help="Race column to color by for tracts (default: auto)")
    ap.add_argument("--max-points", type=int, default=50000, help="Max points before HeatMap for service lines")
    args = ap.parse_args()

    layers = scan_layers(args.data_dir, args.cache_dir)
    if not layers:
        log(f"[ERROR] No files found in {args.data_dir}. Run downloads first.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # Core: neighborhoods
    neighborhoods_path = resolve_by_keywords(layers, ["neighborhood"])
    if neighborhoods_path:
        nbh = load_any(neighborhoods_path)
        if nbh is not None and not nbh.empty:
            name_field = "Name" if "Name" in nbh.columns else ("NAME" if "NAME" in nbh.columns else None)
            map_polygons(nbh, out_html=os.path.join(args.out_dir, "neighborhoods.html"), name_field=name_field)
        else:
            log("[WARN] Neighborhoods layer empty or failed to load.")
    else:
        log("[WARN] Neighborhoods not found.")

    # Core: high school zones
    hs_zones_path = resolve_by_keywords(layers, ["high_school_zones", "highschool", "school_zone", "zone"])
    if hs_zones_path:
        hs = load_any(hs_zones_path)
        if hs is not None and not hs.empty:
            name_field = "Name" if "Name" in hs.columns else None
            map_polygons(hs, out_html=os.path.join(args.out_dir, "high_school_zones.html"), name_field=name_field)
        else:
            log("[WARN] High school zones layer empty or failed to load.")
    else:
        log("[WARN] High school zones not found.")

    # Core: service lines
    servicelines_path = resolve_by_keywords(layers, ["serviceline", "service_line", "richmond_serviceline_view"]) 
    if servicelines_path:
        sl = load_any(servicelines_path)
        if sl is not None and not sl.empty:
            if sl.geometry is None or sl.geometry.isna().all():
                if {"lon", "lat"}.issubset(sl.columns):
                    sl = gpd.GeoDataFrame(sl, geometry=gpd.points_from_xy(sl["lon"], sl["lat"]), crs=4326)
                else:
                    log("[WARN] Service lines has no geometry/coordinates; skipping.")
            map_service_lines(sl, out_html=os.path.join(args.out_dir, "service_lines.html"), max_points=args.max_points)
        else:
            log("[WARN] Service lines layer empty or failed to load.")
    else:
        log("[WARN] Service lines not found.")

    # Core: race/ethnicity tracts — prefer polygons
    race_path = None
    polygon_candidate = None
    for entry in layers.values():
        for path in entry.values():
            g = load_any(path)
            if g is None:
                continue
            cols = g.columns
            has_race_cols = any(c in cols for c in ["Pblack", "Phispanic", "Pwhite", "Pnwhite", "totalE", "WhiteE", "BlackE", "HispanicE", "AsianE", "NativeE"]) 
            if not has_race_cols:
                continue
            try:
                types = g.geom_type.dropna().unique().tolist() if hasattr(g, "geom_type") else []
            except Exception:
                types = []
            if any("Polygon" in t for t in types):
                polygon_candidate = path
                break
            if race_path is None:
                race_path = path
        if polygon_candidate:
            break
    if polygon_candidate:
        race_path = polygon_candidate

    if race_path:
        rc = load_any(race_path)
        if rc is not None and not rc.empty:
            try:
                types = rc.geom_type.dropna().unique().tolist() if hasattr(rc, "geom_type") else []
                log(f"[INFO] Race source: {os.path.basename(race_path)} | geom: {types}")
            except Exception:
                pass
            color_col = args.race_col
            if color_col is None:
                for c in ["Pblack", "Phispanic", "Pwhite", "Pnwhite"]:
                    if c in rc.columns:
                        color_col = c
                        break
            # Boundaries quick view and choropleth
            name_field = "NAME" if "NAME" in rc.columns else None
            map_polygons(rc, out_html=os.path.join(args.out_dir, "race_ethnicity_tracts_boundaries.html"), name_field=name_field, boundaries_only=True)
            map_polygons(rc, out_html=os.path.join(args.out_dir, "race_ethnicity_tracts.html"), name_field=name_field, color_by=color_col)
        else:
            log("[WARN] Race/Ethnicity layer empty or failed to load.")
    else:
        log("[WARN] Race/Ethnicity layer not found.")

    # Enriched neighborhoods (if available)
    enriched_base = os.path.join(args.derived_dir, "neighborhoods_enriched")
    enriched = None
    for ext in (".geoparquet", ".geojson", ".parquet"):
        p = enriched_base + ext
        if os.path.exists(p):
            try:
                enriched = gpd.read_parquet(p) if ext in (".geoparquet", ".parquet") else gpd.read_file(p)
                break
            except Exception as e:
                log(f"[WARN] Failed to read {p}: {e}")
    if enriched is not None and not enriched.empty:
        name_field = None
        for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
            if c in enriched.columns:
                name_field = c
                break
        choropleth_enriched(enriched, args.out_dir, name_field)
    else:
        log("[INFO] Enriched neighborhoods not found; skipping enriched maps.")

    # Index page
    items = []
    for fn in sorted(os.listdir(args.out_dir)):
        if fn.endswith(".html"):
            items.append(fn)
    if items:
        index_path = os.path.join(args.out_dir, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("<h2>Interactive Maps</h2>\n<ul>\n")
            for fn in items:
                f.write(f"  <li><a href='{fn}' target='_blank'>{fn}</a></li>\n")
            f.write("</ul>\n")
        log(f"[OK] Wrote index -> {index_path}")


if __name__ == "__main__":
    main()
