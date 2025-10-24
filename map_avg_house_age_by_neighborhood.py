import os
from typing import Optional, List, Dict

import geopandas as gpd
import pandas as pd
import numpy as np


RAW_DIR = "data_raw"
ANALYSIS_DIR = os.path.join("data", "analysis")
OUT_HTML = os.path.join("outputs", "interactive", "neighborhoods_avg_house_age.html")


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
    for c in ["neighborhood_name", "Name", "NAME", "neighborhood", "Neighborhood"]:
        if c in df.columns:
            return c
    # case-insensitive fallback
    for c in df.columns:
        if str(c).lower() in {"name", "neighborhood", "neighborhood_name"}:
            return c
    return None


def _make_colormap(series: pd.Series):
    from branca.colormap import LinearColormap
    s = pd.to_numeric(series, errors="coerce")
    vmin, vmax = float(s.quantile(0.05)), float(s.quantile(0.95))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(s.min(skipna=True) or 0), float(s.max(skipna=True) or 1)
        if vmin == vmax:
            vmax = vmin + 1
    return LinearColormap(["#f7fbff", "#6baed6", "#08306b"], vmin=vmin, vmax=vmax)


def compute_avg_house_age(csv_path: str, neigh_col_candidates: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if df is None or df.empty:
        raise SystemExit("Servicelines CSV is empty")

    # Determine neighborhood column in CSV
    nb_col = None
    for c in neigh_col_candidates + ["neighborhood_name", "Neighborhood", "neighborhood", "NAME", "Name"]:
        if c in df.columns:
            nb_col = c
            break
    if nb_col is None:
        # case-insensitive
        for c in df.columns:
            if str(c).lower() in {"neighborhood_name", "neighborhood", "name"}:
                nb_col = c
                break
    if nb_col is None:
        raise SystemExit("Could not detect neighborhood column in servicelines CSV")

    # Clean year and compute age
    y = pd.to_numeric(df.get("yearstructbuilt"), errors="coerce")
    current_year = pd.Timestamp.now().year
    valid = (y > 0) & (y <= current_year)
    df = df.loc[valid].copy()
    df["age_years"] = current_year - y[valid]

    # Group by neighborhood
    grp = df.groupby(nb_col, dropna=True).agg(
        avg_house_age=("age_years", "mean"),
        n_houses=("age_years", "count"),
    )
    grp = grp.reset_index().rename(columns={nb_col: "__nbh__"})
    return grp


def map_avg_age(neigh_path: str, imputed_csv: str, out_html: str):
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

    # Compute per-neighborhood average house age from servicelines CSV
    stats = compute_avg_house_age(imputed_csv, neigh_col_candidates=[name_geom])

    # Merge
    merged = nbh.merge(stats, left_on=name_geom, right_on="__nbh__", how="left")

    # Prepare style
    s = pd.to_numeric(merged["avg_house_age"], errors="coerce")
    cmap = _make_colormap(s)
    nodata = "#bdbdbd"

    def style_fn(feat):
        val = feat["properties"].get("avg_house_age")
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
    disp_col = "__disp_avg_age__"
    merged[disp_col] = pd.to_numeric(merged["avg_house_age"], errors="coerce").round(1)
    fields: List[str] = [name_geom, disp_col, "n_houses"]
    aliases: List[str] = ["Neighborhood", "Avg House Age (years)", "Count"]
    tooltip = folium.features.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True)

    # Map
    minx, miny, maxx, maxy = merged.total_bounds
    lat, lon = (miny + maxy) / 2, (minx + maxx) / 2
    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")
    folium.GeoJson(
        merged.to_json(default=str),
        name="Average House Age by Neighborhood",
        style_function=style_fn,
        tooltip=tooltip,
    ).add_to(m)
    try:
        cmap.caption = "Average House Age (years)"
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

    imputed_csv = os.path.join(ANALYSIS_DIR, "servicelines_with_imputed_materials.csv")
    if not os.path.exists(imputed_csv):
        raise SystemExit(f"Imputed servicelines CSV not found: {imputed_csv}")

    map_avg_age(neigh_path, imputed_csv, OUT_HTML)


if __name__ == "__main__":
    main()
