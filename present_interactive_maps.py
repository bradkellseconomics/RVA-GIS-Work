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


def scan_data_dir(data_dir: str) -> Dict[str, Dict[str, str]]:
    layers: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(data_dir):
        return layers
    for fn in os.listdir(data_dir):
        path = os.path.join(data_dir, fn)
        if not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(fn)
        ext = ext.lower()
        if ext in {".geojson", ".geoparquet", ".parquet", ".csv"}:
            layers.setdefault(base.lower(), {})[ext] = path
    return layers


def merge_layers(primary: Dict[str, Dict[str, str]], secondary: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    merged = dict(primary)
    for base, exts in secondary.items():
        if base in merged:
            merged[base].update(exts)
        else:
            merged[base] = dict(exts)
    return merged


def resolve_by_keywords(layers: Dict[str, Dict[str, str]], keywords: List[str]) -> Optional[str]:
    # Prefer by base name, then by filename
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
        # fallback
        return gpd.read_file(path)
    except Exception as e:
        log(f"[WARN] Failed to read {path}: {e}")
        return None


def center_from_bounds(gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    try:
        g = ensure_wgs84(gdf)
        minx, miny, maxx, maxy = g.total_bounds
        return ( (miny + maxy) / 2, (minx + maxx) / 2 )
    except Exception:
        # Richmond approx
        return (37.5407, -77.4360)


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

    # If many points, render as heatmap for performance; else clustered markers
    if len(g) > max_points:
        coords = list(zip(g.geometry.y, g.geometry.x))
        HeatMap(coords, radius=6, blur=8, max_zoom=13).add_to(m)
        folium.map.LayerControl().add_to(m)
        log(f"[OK] HeatMap with {len(coords)} points")
    else:
        mc = MarkerCluster().add_to(m)
        # Add basic popup fields if available
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
        log(f"[OK] Clustered {len(g)} points")

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    log(f"[OK] Wrote map -> {out_html}")


def _make_colormap(gdf: gpd.GeoDataFrame, column: str):
    from branca.colormap import LinearColormap
    s = pd.to_numeric(gdf[column], errors="coerce")
    vmin, vmax = float(s.quantile(0.05)), float(s.quantile(0.95))
    if vmin == vmax:
        vmin, vmax = float(s.min()), float(s.max())
    return LinearColormap(["#fee8c8", "#fdbb84", "#e34a33"], vmin=vmin, vmax=vmax)


def map_polygons(gdf: gpd.GeoDataFrame, out_html: str, name_field: Optional[str] = None, color_by: Optional[str] = None):
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

    tooltip_fields: List[str] = []
    if name_field and name_field in g.columns:
        tooltip_fields.append(name_field)

    style_function = None
    highlight = folium.features.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None

    if color_by and color_by in g.columns:
        cmap = _make_colormap(g, color_by)

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
    else:
        def style_fn_simple(_):
            return {"fillColor": "#9ecae1", "color": "#6baed6", "weight": 0.8, "fillOpacity": 0.6}

        style_function = style_fn_simple

    folium.GeoJson(
        g.to_json(default=str),
        name=os.path.basename(out_html),
        style_function=style_function,
        tooltip=highlight,
    ).add_to(m)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    log(f"[OK] Wrote map -> {out_html}")


def main():
    ap = argparse.ArgumentParser(description="Create interactive HTML maps for core datasets.")
    ap.add_argument("--data-dir", default="data_raw", help="Folder with raw files")
    ap.add_argument("--out-dir", default=os.path.join("outputs", "interactive"), help="Folder for HTML outputs")
    ap.add_argument("--max-points", type=int, default=50000, help="Max points before switching to HeatMap for service lines")
    ap.add_argument("--race-col", default=None, help="Column to color by for race/ethnicity (default: auto Pblack/Phispanic/Pwhite)")
    args = ap.parse_args()

    # Scan primary raw dir and also optional cache dir (rva_layers) for polygon sources
    layers = scan_data_dir(args.data_dir)
    if os.path.isdir("rva_layers"):
        layers = merge_layers(layers, scan_data_dir("rva_layers"))
    if not layers:
        log(f"[ERROR] No files found in {args.data_dir}. Run downloads first.")
        sys.exit(1)

    # Resolve expected datasets
    neighborhoods_path = resolve_by_keywords(layers, ["neighborhood"])
    hs_zones_path = resolve_by_keywords(layers, ["high_school_zones", "highschool", "school_zone", "zone"])
    servicelines_path = resolve_by_keywords(layers, ["serviceline", "service_line", "richmond_serviceline_view"]) 

    race_path = None
    # Prefer polygonal race layer if available; otherwise fallback to any with race columns
    def has_race_cols(g: gpd.GeoDataFrame) -> bool:
        cols = g.columns
        return any(c in cols for c in ["Pblack", "Phispanic", "Pwhite", "Pnwhite", "totalE", "WhiteE", "BlackE", "HispanicE", "AsianE", "NativeE"]) 

    polygon_candidate = None
    for entry in layers.values():
        for path in entry.values():
            g = load_any(path)
            if g is None:
                continue
            if not has_race_cols(g):
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

    # Neighborhoods
    if neighborhoods_path:
        nbh = load_any(neighborhoods_path)
        if nbh is not None and not nbh.empty:
            name_field = "Name" if "Name" in nbh.columns else ("NAME" if "NAME" in nbh.columns else None)
            map_polygons(nbh, out_html=os.path.join(args.out_dir, "neighborhoods.html"), name_field=name_field)
        else:
            log("[WARN] Neighborhoods layer empty or failed to load.")
    else:
        log("[WARN] Neighborhoods not found.")

    # High school zones
    if hs_zones_path:
        hs = load_any(hs_zones_path)
        if hs is not None and not hs.empty:
            name_field = "Name" if "Name" in hs.columns else None
            map_polygons(hs, out_html=os.path.join(args.out_dir, "high_school_zones.html"), name_field=name_field)
        else:
            log("[WARN] High school zones layer empty or failed to load.")
    else:
        log("[WARN] High school zones not found.")

    # Service lines
    if servicelines_path:
        sl = load_any(servicelines_path)
        if sl is not None and not sl.empty:
            # Ensure geometry
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

    # Race/Ethnicity tracts choropleth
    if race_path:
        rc = load_any(race_path)
        if rc is not None and not rc.empty:
            # Report chosen source and geometry type
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
            map_polygons(rc, out_html=os.path.join(args.out_dir, "race_ethnicity_tracts.html"), name_field="NAME" if "NAME" in rc.columns else None, color_by=color_col)
        else:
            log("[WARN] Race/Ethnicity layer empty or failed to load.")
    else:
        log("[WARN] Race/Ethnicity layer not found.")


if __name__ == "__main__":
    main()
