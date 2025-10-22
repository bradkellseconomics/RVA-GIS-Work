import os
import sys
from typing import Dict, Optional, List

import geopandas as gpd
import pandas as pd


RAW_DIR = "data_raw"
OUT_DIR = os.path.join("data", "analysis")


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    try:
        if gdf.crs is None:
            return gdf.set_crs(epsg=epsg)
        return gdf.to_crs(epsg=epsg)
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


def resolve_path(layers: Dict[str, Dict[str, str]], keywords: List[str]) -> Optional[str]:
    # Prefer Parquet -> GeoJSON -> CSV
    for key, entry in layers.items():
        if any(kw in key for kw in keywords):
            for ext in (".geoparquet", ".parquet", ".geojson", ".csv"):
                if ext in entry:
                    return entry[ext]
    for entry in layers.values():
        for path in entry.values():
            if any(kw in os.path.basename(path).lower() for kw in keywords):
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


def pick_name_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
        if c in gdf.columns:
            return c
    for c in gdf.columns:
        if pd.api.types.is_object_dtype(gdf[c]):
            return c
    return None


def normalize_status(val) -> str:
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().lower()
    if "non" in s and "lead" in s:
        return "Non-Lead"
    if "lead" in s:
        return "Lead"
    if s in {"yes", "y"}:
        return "Lead"
    if s in {"no", "n"}:
        return "Non-Lead"
    return "Unknown"


def compute_lead_by_neighborhood(sl: gpd.GeoDataFrame, nbh: gpd.GeoDataFrame, name_col: str) -> pd.DataFrame:
    pts = sl.copy()
    src_col = None
    for c in ["bothsidesstatus", "utilmaterial", "custmaterial", "everlead"]:
        if c in pts.columns:
            src_col = c
            break
    pts["status_norm"] = pts[src_col].map(normalize_status) if src_col else "Unknown"

    pts_5070 = ensure_crs(pts, 5070)
    nbh_5070 = ensure_crs(nbh, 5070)
    joined = gpd.sjoin(pts_5070, nbh_5070[[name_col, "geometry"]], how="left", predicate="within")

    grp = joined.groupby(name_col)["status_norm"].value_counts().unstack(fill_value=0)
    for c in ["Lead", "Non-Lead", "Unknown"]:
        if c not in grp.columns:
            grp[c] = 0
    grp["total_points"] = grp[["Lead", "Non-Lead", "Unknown"]].sum(axis=1)
    grp["known_points"] = grp[["Lead", "Non-Lead"]].sum(axis=1)
    grp["pct_lead_total"] = (grp["Lead"] / grp["total_points"]).where(grp["total_points"] > 0, 0.0)
    grp["pct_nonlead_total"] = (grp["Non-Lead"] / grp["total_points"]).where(grp["total_points"] > 0, 0.0)
    grp["pct_unknown"] = (grp["Unknown"] / grp["total_points"]).where(grp["total_points"] > 0, 0.0)
    # Also provide rates among known
    grp["pct_lead_known"] = (grp["Lead"] / grp["known_points"]).where(grp["known_points"] > 0, 0.0)
    grp["pct_nonlead_known"] = (grp["Non-Lead"] / grp["known_points"]).where(grp["known_points"] > 0, 0.0)

    # House age/Year built cleaning and aggregation
    year_col = None
    for c in ["yearstructbuilt", "YearBuilt", "year_built", "YEARBUILT"]:
        if c in joined.columns:
            year_col = c
            break
    if year_col:
        years = pd.to_numeric(joined[year_col], errors="coerce")
        current_year = pd.Timestamp.now().year
        # Consider reasonable range
        valid = years.between(1800, current_year)
        joined_valid = joined.loc[valid].copy()
        joined_valid["_year"] = years[valid]
        age_stats = joined_valid.groupby(name_col)["_year"].agg(["mean", "count"]).rename(columns={"mean": "avg_year_built", "count": "count_year_samples"})
        age_stats["avg_house_age"] = current_year - age_stats["avg_year_built"]
        grp = grp.join(age_stats, how="left")

    return grp.reset_index()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    layers = scan_dir(RAW_DIR)
    if not layers:
        log(f"[ERROR] No files in {RAW_DIR}. Run downloads first.")
        sys.exit(1)

    sl_path = resolve_path(layers, ["serviceline", "service_line", "richmond_serviceline_view"]) 
    nbh_path = resolve_path(layers, ["neighborhood"]) 
    if not sl_path or not nbh_path:
        log("[ERROR] Missing service lines or neighborhoods in data_raw/")
        sys.exit(1)

    sl = load_any(sl_path)
    nbh = load_any(nbh_path)
    if sl is None or sl.empty:
        log("[ERROR] Service lines empty.")
        sys.exit(1)
    if nbh is None or nbh.empty:
        log("[ERROR] Neighborhoods empty.")
        sys.exit(1)

    name_col = pick_name_col(nbh)
    if not name_col:
        log("[ERROR] Could not detect neighborhood name column.")
        sys.exit(1)

    lead_by_nbh = compute_lead_by_neighborhood(sl, nbh, name_col)
    out_csv = os.path.join(OUT_DIR, "neighborhoods_lead_summary.csv")
    lead_by_nbh.to_csv(out_csv, index=False)
    log(f"[OK] Wrote {out_csv}")


if __name__ == "__main__":
    main()
