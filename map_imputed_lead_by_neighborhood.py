import os
from typing import Optional, List, Dict

import geopandas as gpd
import pandas as pd


RAW_DIR = "data_raw"
ANALYSIS_DIR = os.path.join("data", "analysis")
OUT_HTML = os.path.join("outputs", "interactive", "neighborhoods_imputed_lead.html")


def log(msg: str) -> None:
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
        if ext.lower() in {".geojson", ".geoparquet", ".parquet", ".csv"}:
            layers.setdefault(base.lower(), {})[ext.lower()] = p
    return layers


def resolve_neighborhoods_path(layers: Dict[str, Dict[str, str]]) -> Optional[str]:
    # Prefer geoparquet -> geojson -> others
    for key, entry in layers.items():
        if "neighborhood" in key:
            for ext in (".geoparquet", ".parquet", ".geojson", ".csv"):
                if ext in entry:
                    return entry[ext]
    for entry in layers.values():
        for path in entry.values():
            if "neighborhood" in os.path.basename(path).lower():
                return path
    return None


def load_any(path: str) -> gpd.GeoDataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".geoparquet", ".parquet"}:
        return gpd.read_parquet(path)
    if ext == ".geojson":
        return gpd.read_file(path)
    if ext == ".csv":
        df = pd.read_csv(path)
        return gpd.GeoDataFrame(df, geometry=None)
    return gpd.read_file(path)


def pick_name_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
        if c in df.columns:
            return c
    # case-insensitive fallback
    for c in df.columns:
        if str(c).lower() in {"name", "neighborhood", "neighborhood_name"}:
            return c
    return None


def pick_imputed_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    # percent column
    pct = None
    for c in df.columns:
        lc = str(c).lower()
        if ("pct" in lc or "percent" in lc) and ("lead" in lc):
            pct = c
            break
    # total count and total lead
    total = None
    for c in ["total_count", "n_total", "count", "total" , "pipes_total"]:
        if c in df.columns:
            total = c
            break
    total_lead = None
    for c in ["total_lead", "n_lead", "lead_count", "pipes_lead"]:
        if c in df.columns:
            total_lead = c
            break
    return {"pct": pct, "total": total, "total_lead": total_lead}


def _make_colormap(series: pd.Series):
    from branca.colormap import LinearColormap
    s = pd.to_numeric(series, errors="coerce")
    vmin, vmax = float(s.quantile(0.05)), float(s.quantile(0.95))
    if vmin == vmax:
        vmin, vmax = float(s.min()), float(s.max())
    return LinearColormap(["#fee8c8", "#fdbb84", "#e34a33"], vmin=vmin, vmax=vmax)


def map_imputed(neigh_path: str, imputed_csv: str, out_html: str):
    try:
        import folium
    except ImportError:
        log("[ERROR] folium is required. Install with: pip install folium")
        return

    nbh = load_any(neigh_path)
    if nbh is None or nbh.empty:
        raise SystemExit("Neighborhoods layer failed to load or is empty")
    nbh = ensure_wgs84(nbh)
    name_geom = pick_name_col(nbh)
    if not name_geom:
        raise SystemExit("Could not detect neighborhood name column in neighborhoods layer")

    df = pd.read_csv(imputed_csv)
    if df is None or df.empty:
        raise SystemExit("Imputed summary CSV is empty")
    name_imp = pick_name_col(df)
    if not name_imp:
        raise SystemExit("Could not detect neighborhood name column in imputed summary CSV")
    cols = pick_imputed_cols(df)
    if not cols["pct"]:
        raise SystemExit("Could not find an imputed percent lead column in summary CSV")

    # Merge
    merged = nbh.merge(df, left_on=name_geom, right_on=name_imp, how="left")

    # Prepare tooltip fields
    pct_col = cols["pct"]
    total_col = cols["total"]
    total_lead_col = cols["total_lead"]

    # Make a display percent column
    disp_col = f"__disp_{pct_col}"
    s = pd.to_numeric(merged[pct_col], errors="coerce")
    # If likely proportions (<= 1), show as 0-100%
    if s.dropna().quantile(0.95) <= 1.0:
        merged[disp_col] = (s * 100).round(1).astype(str) + "%"
    else:
        merged[disp_col] = s.round(1).astype(str) + "%"

    # Style function
    nodata = "#bdbdbd"
    cmap = _make_colormap(s)

    def style_fn(feat):
        val = feat["properties"].get(pct_col)
        try:
            v = float(val) if val is not None else None
        except Exception:
            v = None
        if v is None or pd.isna(v):
            fill = nodata
        else:
            try:
                fill = cmap(v)
            except Exception:
                fill = nodata
        return {"fillColor": fill, "color": "#666", "weight": 0.8, "fillOpacity": 0.7}

    # Tooltip
    fields: List[str] = [name_geom, disp_col]
    aliases: List[str] = ["Neighborhood", "Imputed % Lead"]
    if total_col and total_col in merged.columns:
        fields.append(total_col); aliases.append("Total Pipes")
    if total_lead_col and total_lead_col in merged.columns:
        fields.append(total_lead_col); aliases.append("Total Lead")
    tooltip = folium.features.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True)

    # Map
    minx, miny, maxx, maxy = merged.total_bounds
    lat, lon = (miny + maxy) / 2, (minx + maxx) / 2
    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(
        merged.to_json(default=str),
        name="Imputed Lead by Neighborhood",
        style_function=style_fn,
        tooltip=tooltip,
    ).add_to(m)
    try:
        cmap.caption = pct_col
        cmap.add_to(m)
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    log(f"[OK] Wrote map -> {out_html}")


def main():
    layers = scan_dir(RAW_DIR)
    neigh_path = resolve_neighborhoods_path(layers)
    if not neigh_path:
        raise SystemExit("Neighborhoods geometry not found in data_raw/")

    imputed_csv = os.path.join(ANALYSIS_DIR, "neighborhoods_lead_imputed_summary.csv")
    if not os.path.exists(imputed_csv):
        raise SystemExit(f"Imputed summary not found: {imputed_csv}")

    map_imputed(neigh_path, imputed_csv, OUT_HTML)


if __name__ == "__main__":
    main()

