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
        ext = ext.lower()
        if ext in {".geojson", ".geoparquet", ".parquet", ".csv"}:
            layers.setdefault(base.lower(), {})[ext] = p
    return layers


def resolve_path(layers: Dict[str, Dict[str, str]], keywords: List[str]) -> Optional[str]:
    # prefer Parquet -> GeoJSON -> CSV
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


def resolve_polygon_path(layers: Dict[str, Dict[str, str]], keywords: List[str]) -> Optional[str]:
    polygon = None
    any_match = None
    for entry in layers.values():
        for path in entry.values():
            name = os.path.basename(path).lower()
            if any(kw in name for kw in keywords):
                g = load_any(path)
                if g is None:
                    continue
                try:
                    types = g.geom_type.dropna().unique().tolist() if hasattr(g, "geom_type") else []
                except Exception:
                    types = []
                if any("Polygon" in t for t in types):
                    polygon = path
                    break
                if any_match is None:
                    any_match = path
        if polygon:
            break
    return polygon or any_match


def pick_race_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    cols: List[str] = []
    for c in ["Pwhite", "Pblack", "Phispanic", "Pasian", "Pnative", "PHIPI"]:
        if c in gdf.columns and pd.api.types.is_numeric_dtype(gdf[c]):
            cols.append(c)
    # Fallback to any P* numeric columns
    if not cols:
        for c in gdf.columns:
            if isinstance(c, str) and c.startswith("P") and pd.api.types.is_numeric_dtype(gdf[c]):
                cols.append(c)
    return cols[:10]


def pick_income_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    cols: List[str] = []
    for c in gdf.columns:
        if pd.api.types.is_numeric_dtype(gdf[c]):
            lc = str(c).lower()
            if ("income" in lc) or ("median" in lc) or c in {"MHI", "MEDHHINC"}:
                cols.append(c)
    return cols[:10]


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    layers = scan_dir(RAW_DIR)
    if not layers:
        log(f"[ERROR] No files in {RAW_DIR}. Run downloads first.")
        sys.exit(1)

    sl_path = resolve_path(layers, ["serviceline", "service_line", "richmond_serviceline_view"]) 
    nbh_path = resolve_path(layers, ["neighborhood"]) 
    # Optional tract layers for direct house-level enrichment
    race_tract_path = resolve_polygon_path(layers, ["race", "ethnicity", "rva_race", "RVA_Race_and_Ethnicity_2020"])
    income_tract_path = resolve_polygon_path(layers, ["income"])

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

    # Spatial join service lines to neighborhoods to add neighborhood name
    sl_wgs = ensure_crs(sl, 4326)
    nbh_wgs = ensure_crs(nbh, 4326)
    try:
        sl_joined = gpd.sjoin(sl_wgs, nbh_wgs[[name_col, "geometry"]], how="left", predicate="within")
    except Exception:
        sl_joined = gpd.sjoin(sl_wgs, nbh_wgs[[name_col, "geometry"]], how="left")

    sl_enriched = sl_joined.copy()

    # Directly join race tracts to houses (replace neighborhood-based race)
    if race_tract_path:
        race_gdf = load_any(race_tract_path)
        if race_gdf is not None and not race_gdf.empty:
            race_gdf = ensure_crs(race_gdf, 4326)
            rcols = pick_race_columns(race_gdf)
            # Keep GEOID/NAME if present for reference
            keep_meta = [c for c in ["GEOID", "NAME"] if c in race_gdf.columns]
            cols_to_keep = ["geometry"] + keep_meta + rcols
            try:
                race_slim = race_gdf[cols_to_keep]
            except Exception:
                race_slim = race_gdf
            try:
                joined_race = gpd.sjoin(sl_enriched, race_slim, how="left", predicate="within", rsuffix="_race")
            except Exception:
                joined_race = gpd.sjoin(sl_enriched, race_slim, how="left", rsuffix="_race")
            sl_enriched = joined_race
            log(f"[INFO] Joined race tracts to houses with columns: {', '.join(rcols[:5])}{'…' if len(rcols)>5 else ''}")
        else:
            log("[INFO] Race tracts layer failed to load; skipping race enrich.")
    else:
        log("[INFO] Race tracts not found; skipping race enrich.")

    # Directly join income tracts to houses
    if income_tract_path:
        inc_gdf = load_any(income_tract_path)
        if inc_gdf is not None and not inc_gdf.empty:
            inc_gdf = ensure_crs(inc_gdf, 4326)
            icols = pick_income_columns(inc_gdf)
            keep_meta = [c for c in ["GEOID", "NAME"] if c in inc_gdf.columns]
            cols_to_keep = ["geometry"] + keep_meta + icols
            try:
                inc_slim = inc_gdf[cols_to_keep]
            except Exception:
                inc_slim = inc_gdf
            try:
                joined_inc = gpd.sjoin(sl_enriched, inc_slim, how="left", predicate="within", rsuffix="_inc")
            except Exception:
                joined_inc = gpd.sjoin(sl_enriched, inc_slim, how="left", rsuffix="_inc")
            sl_enriched = joined_inc
            log(f"[INFO] Joined income tracts to houses with columns: {', '.join(icols[:5])}{'…' if len(icols)>5 else ''}")
        else:
            log("[INFO] Income tracts layer failed to load; skipping income enrich.")
    else:
        log("[INFO] Income tracts not found; skipping income enrich.")

    # Save house-level outputs
    out_house_csv = os.path.join(OUT_DIR, "servicelines_house_with_attributes.csv")
    out_house_pq = os.path.join(OUT_DIR, "servicelines_house_with_attributes.geoparquet")

    try:
        sl_enriched.to_parquet(out_house_pq, index=False)
        log(f"[OK] Wrote {out_house_pq}")
    except Exception as e:
        log(f"[WARN] Failed to write GeoParquet: {e}")

    try:
        df_flat = pd.DataFrame(sl_enriched)
        if "geometry" in df_flat.columns:
            df_flat = df_flat.drop(columns=["geometry"])  # type: ignore[arg-type]
        df_flat.to_csv(out_house_csv, index=False)
        log(f"[OK] Wrote {out_house_csv}")
    except Exception as e:
        log(f"[WARN] Failed to write CSV: {e}")


if __name__ == "__main__":
    main()
