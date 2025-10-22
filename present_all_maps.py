import argparse
import os
import sys
from typing import Dict, Optional, Tuple, List

import geopandas as gpd
import numpy as np
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


def _make_colormap_blues(series: pd.Series):
    from branca.colormap import LinearColormap
    s = pd.to_numeric(series, errors="coerce")
    vmin, vmax = float(s.quantile(0.05)), float(s.quantile(0.95))
    if vmin == vmax:
        vmin, vmax = float(s.min()), float(s.max())
    # Light to dark blues
    return LinearColormap(["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"], vmin=vmin, vmax=vmax)


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

    # Build tooltip: always include name if available, and include the themed column if provided
    tooltip = None
    fields: List[str] = []
    aliases: List[str] = []
    if name_field and name_field in g.columns:
        fields.append(name_field)
        aliases.append("Neighborhood")
    # Decide which column to show in tooltip (may be formatted differently from the color_by numeric data)
    display_col = None
    if (not boundaries_only) and color_by and color_by in g.columns and pd.api.types.is_numeric_dtype(g[color_by]):
        display_col = color_by
        # Special formatting: present years/ages as whole numbers (no thousands separators)
        if color_by in ["avg_year_built", "avg_house_age"]:
            disp_name = f"__disp_{color_by}"
            try:
                g[disp_name] = pd.to_numeric(g[color_by], errors="coerce").round(0).astype("Int64").astype(str)
                display_col = disp_name
            except Exception:
                display_col = color_by
        fields.append(display_col)
        aliases.append(color_by.replace("pct_", "Pct ").replace("_", " ").title())
    if fields:
        tooltip = folium.features.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True)  # type: ignore[arg-type]

    style_function = (lambda _: {"fill": False, "color": "#444", "weight": 1.0}) if boundaries_only else None

    nodata_color = "#bdbdbd"  # medium grey for missing values
    if not boundaries_only and color_by and color_by in g.columns:
        cmap = _make_colormap(g[color_by])

        def style_fn(feat):
            val = feat["properties"].get(color_by)
            try:
                vnum = float(val) if val is not None else None
            except Exception:
                vnum = None
            if vnum is None or (isinstance(vnum, float) and (np.isnan(vnum))):
                col = nodata_color
            else:
                try:
                    col = cmap(vnum)
                except Exception:
                    col = nodata_color
            return {"fillColor": col, "color": "#666666", "weight": 0.8, "fillOpacity": 0.7}

        style_function = style_fn
        cmap.caption = color_by
        cmap.add_to(m)
        # Add a small legend entry for no data (grey)
        try:
            no_data_html = (
                f"""
                <div style=\"position: fixed; bottom: 10px; left: 10px; z-index: 9999;\
                            background: white; padding: 6px 8px; border:1px solid #bbb;\">
                  <div style=\"font-size:12px;\"><span style=\"display:inline-block;width:12px;height:12px;\
                      background:{nodata_color};margin-right:6px;border:1px solid #999;\"></span>No data</div>
                </div>
                """
            )
            m.get_root().html.add_child(folium.Element(no_data_html))
        except Exception:
            pass

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


def map_bivariate_polygons(
    gdf: gpd.GeoDataFrame,
    out_html: str,
    name_field: Optional[str],
    income_col: str,
    race_col: str,
):
    try:
        import folium
    except ImportError:
        log("[ERROR] folium is required. Install with: pip install folium")
        return

    g = ensure_wgs84(gdf).dropna(subset=["geometry"])  # type: ignore[arg-type]
    if len(g) == 0 or income_col not in g.columns or race_col not in g.columns:
        log("[WARN] Bivariate map skipped due to missing columns or empty data.")
        return

    lat, lon = center_from_bounds(g)
    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")

    # Tooltips show both income and race
    fields: List[str] = []
    aliases: List[str] = []
    if name_field and name_field in g.columns:
        fields.append(name_field)
        aliases.append("Neighborhood")
    fields.extend([income_col, race_col])
    aliases.extend([
        income_col.replace("pct_", "Pct ").replace("_", " ").title(),
        race_col.replace("pct_", "Pct ").replace("_", " ").title(),
    ])
    tooltip = folium.features.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True)  # type: ignore[arg-type]

    nodata_fill = "#bdbdbd"
    nodata_stroke = "#969696"
    income_cmap = _make_colormap(g[income_col])
    race_cmap = _make_colormap_blues(g[race_col])

    def style_fn(feat):
        inc = feat["properties"].get(income_col)
        rac = feat["properties"].get(race_col)
        # Fill by income, border by race
        try:
            incv = float(inc) if inc is not None else None
        except Exception:
            incv = None
        try:
            racv = float(rac) if rac is not None else None
        except Exception:
            racv = None
        if incv is None or (isinstance(incv, float) and np.isnan(incv)):
            fill = nodata_fill
        else:
            try:
                fill = income_cmap(incv)
            except Exception:
                fill = nodata_fill
        if racv is None or (isinstance(racv, float) and np.isnan(racv)):
            stroke = nodata_stroke
        else:
            try:
                stroke = race_cmap(racv)
            except Exception:
                stroke = nodata_stroke
        return {"fillColor": fill, "color": stroke, "weight": 1.2, "fillOpacity": 0.7}

    folium.GeoJson(
        g.to_json(default=str),
        name=os.path.basename(out_html),
        style_function=style_fn,
        tooltip=tooltip,
    ).add_to(m)

    # Add legends for both scales
    try:
        income_cmap.caption = income_col
        income_cmap.add_to(m)
        race_cmap.caption = race_col
        race_cmap.add_to(m)
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    log(f"[OK] Wrote map -> {out_html}")


def choropleth_enriched(gdf: gpd.GeoDataFrame, out_dir: str, name_field: Optional[str]):
    # Lead metrics
    for c in ["pct_known_lead", "pct_unknown", "pct_known_nonlead", "pct_known_lead_total", "pct_known_nonlead_total"]:
        if c in gdf.columns:
            map_polygons(gdf, os.path.join(out_dir, f"neighborhoods_enriched_{c}.html"), name_field=name_field, color_by=c)


def pick_income_columns(df: gpd.GeoDataFrame, name_field: Optional[str]) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if name_field and c == name_field:
            continue
        if c == "geometry":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            lc = str(c).lower()
            if ("income" in lc) or ("median" in lc) or c in {"MHI", "MEDHHINC"}:
                cols.append(c)
    # Fallback to any numeric if none matched; limit to first 5
    if not cols:
        cols = [c for c in df.columns if c != "geometry" and (not name_field or c != name_field) and pd.api.types.is_numeric_dtype(df[c])][:5]
    return cols[:6]
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

    # Income by census tract (polygon choropleths)
    income_tract_path = None
    income_polygon_candidate = None
    for entry in layers.values():
        for path in entry.values():
            g = load_any(path)
            if g is None:
                continue
            # Heuristic: presence of an income/median numeric column
            cols = g.columns
            has_income_col = any((isinstance(c, str) and ("income" in c.lower() or "median" in c.lower())) and pd.api.types.is_numeric_dtype(g[c]) for c in cols)
            if not has_income_col:
                continue
            try:
                types = g.geom_type.dropna().unique().tolist() if hasattr(g, "geom_type") else []
            except Exception:
                types = []
            if any("Polygon" in t for t in types):
                income_polygon_candidate = path
                break
            if income_tract_path is None:
                income_tract_path = path
        if income_polygon_candidate:
            break
    if income_polygon_candidate:
        income_tract_path = income_polygon_candidate

    if income_tract_path:
        igdf = load_any(income_tract_path)
        if igdf is not None and not igdf.empty:
            try:
                types = igdf.geom_type.dropna().unique().tolist() if hasattr(igdf, "geom_type") else []
                log(f"[INFO] Income source: {os.path.basename(income_tract_path)} | geom: {types}")
            except Exception:
                pass
            # Determine tract name label if any
            name_field = None
            for c in ["NAME", "Name", "GEOID", "TRACT", "TRACTCE", "GEOID10", "GEOID20"]:
                if c in igdf.columns:
                    name_field = c
                    break
            # Pick income columns and render tract-level maps
            income_cols = pick_income_columns(igdf, name_field)
            if income_cols:
                for c in income_cols:
                    map_polygons(igdf, os.path.join(args.out_dir, f"income_tracts_{c}.html"), name_field=name_field, color_by=c)
            else:
                log("[INFO] No numeric income columns found on tracts to map.")
        else:
            log("[WARN] Income tracts layer empty or failed to load.")
    else:
        log("[INFO] Income tracts not found; skipping tract-level income maps.")

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

    # Income by neighborhood (if available)
    income_pq = os.path.join(args.derived_dir, "..", "analysis", "neighborhoods_income_summary.geoparquet")
    income_csv = os.path.join("data", "analysis", "neighborhoods_income_summary.csv")
    income_gdf = None
    if os.path.exists(income_pq):
        try:
            income_gdf = gpd.read_parquet(income_pq)
        except Exception as e:
            log(f"[WARN] Failed to read {income_pq}: {e}")
    elif os.path.exists(income_csv):
        # Merge CSV with neighborhoods geometry
        if neighborhoods_path:
            try:
                df = pd.read_csv(income_csv)
                nbh = load_any(neighborhoods_path)
                if nbh is not None and not nbh.empty:
                    name_field = None
                    for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
                        if c in nbh.columns:
                            name_field = c
                            break
                    if name_field and name_field in df.columns:
                        income_gdf = ensure_wgs84(nbh).merge(df, on=name_field, how="left")
            except Exception as e:
                log(f"[WARN] Failed to join income CSV to neighborhoods: {e}")

    if income_gdf is not None and not income_gdf.empty:
        # Determine name field again (from merged frame)
        name_field = None
        for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
            if c in income_gdf.columns:
                name_field = c
                break
        cols = pick_income_columns(income_gdf, name_field)
        if cols:
            for c in cols:
                map_polygons(income_gdf, os.path.join(args.out_dir, f"neighborhoods_income_{c}.html"), name_field=name_field, color_by=c)
        else:
            log("[INFO] No numeric income columns found to map.")
    else:
        log("[INFO] neighborhoods_income_summary not found; skipping income maps.")

    # Lead by neighborhood (CSV from builder)
    lead_csv = os.path.join("data", "analysis", "neighborhoods_lead_summary.csv")
    if os.path.exists(lead_csv) and neighborhoods_path:
        try:
            df = pd.read_csv(lead_csv)
            nbh = load_any(neighborhoods_path)
            if nbh is not None and not nbh.empty:
                name_field = None
                for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
                    if c in nbh.columns and c in df.columns:
                        name_field = c
                        break
                if not name_field:
                    # try case-insensitive match
                    for nc in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
                        for dc in df.columns:
                            if dc.lower() == nc.lower() and nc in nbh.columns:
                                df = df.rename(columns={dc: nc}); name_field = nc; break
                        if name_field:
                            break
                if name_field:
                    lead_gdf = ensure_wgs84(nbh).merge(df, on=name_field, how="left")
                    # Map desired columns
                    for c in ["pct_lead_total", "pct_nonlead_total", "pct_unknown", "pct_lead_known", "pct_nonlead_known", "avg_house_age", "avg_year_built"]:
                        if c in lead_gdf.columns:
                            map_polygons(lead_gdf, os.path.join(args.out_dir, f"neighborhoods_lead_{c}.html"), name_field=name_field, color_by=c)
        except Exception as e:
            log(f"[WARN] Failed to visualize lead by neighborhood: {e}")

    # Visualize neighborhood summaries built from matches (race and income)
    matches_race_pq = os.path.join("data_derived", "neighborhoods_from_matches_race.geoparquet")
    matches_income_pq = os.path.join("data_derived", "neighborhoods_from_matches_income.geoparquet")
    # Helper to load PQ or merge CSV with neighborhoods
    def load_matches_layer(pq_path: str, csv_path: str):
        if os.path.exists(pq_path):
            try:
                return gpd.read_parquet(pq_path)
            except Exception as e:
                log(f"[WARN] Failed to read {pq_path}: {e}")
        if os.path.exists(csv_path) and neighborhoods_path:
            try:
                df = pd.read_csv(csv_path)
                nbh = load_any(neighborhoods_path)
                if nbh is not None and not nbh.empty:
                    name_field = None
                    for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
                        if c in nbh.columns:
                            name_field = c
                            break
                    if name_field and name_field in df.columns:
                        return ensure_wgs84(nbh).merge(df, on=name_field, how="left")
            except Exception as e:
                log(f"[WARN] Failed to load matches CSV {csv_path}: {e}")
        return None

    race_from_matches = load_matches_layer(matches_race_pq, os.path.join("data_derived", "neighborhoods_from_matches_race.csv"))
    if race_from_matches is not None and not race_from_matches.empty:
        name_field = None
        for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
            if c in race_from_matches.columns:
                name_field = c
                break
        for c in ["pct_black", "pct_white", "pct_hispanic", "pct_asian", "pct_native", "pct_hipi"]:
            if c in race_from_matches.columns:
                map_polygons(race_from_matches, os.path.join(args.out_dir, f"neighborhoods_matches_{c}.html"), name_field=name_field, color_by=c)

    income_from_matches = load_matches_layer(matches_income_pq, os.path.join("data_derived", "neighborhoods_from_matches_income.csv"))
    if income_from_matches is not None and not income_from_matches.empty:
        name_field = None
        for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
            if c in income_from_matches.columns:
                name_field = c
                break
        cols = pick_income_columns(income_from_matches, name_field)
        for c in cols:
            map_polygons(income_from_matches, os.path.join(args.out_dir, f"neighborhoods_matches_income_{c}.html"), name_field=name_field, color_by=c)

    # Bivariate: race + income together from combined file
    combined_pq = os.path.join("data_derived", "neighborhoods_from_matches_combined.geoparquet")
    combined = None
    if os.path.exists(combined_pq):
        try:
            combined = gpd.read_parquet(combined_pq)
        except Exception as e:
            log(f"[WARN] Failed to read {combined_pq}: {e}")
    if combined is not None and not combined.empty:
        name_field = None
        for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
            if c in combined.columns:
                name_field = c
                break
        # Choose defaults: income col (first) and race col (pct_black preferred)
        inc_cols = pick_income_columns(combined, name_field)
        race_cols = [c for c in ["pct_black", "pct_white", "pct_hispanic", "pct_asian", "pct_native", "pct_hipi"] if c in combined.columns]
        if inc_cols and race_cols:
            income_col = inc_cols[0]
            race_col = "pct_black" if "pct_black" in race_cols else race_cols[0]
            map_bivariate_polygons(combined, os.path.join(args.out_dir, f"neighborhoods_matches_bivariate_{income_col}_{race_col}.html"), name_field=name_field, income_col=income_col, race_col=race_col)

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
