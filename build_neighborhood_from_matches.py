import os
import sys
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd


RAW_DIR = "data_raw"
DERIVED_DIR = "data_derived"


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(epsg=epsg)
    return gdf.to_crs(epsg=epsg)


def scan_dir(dir_path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(dir_path):
        return out
    for fn in os.listdir(dir_path):
        p = os.path.join(dir_path, fn)
        if not os.path.isfile(p):
            continue
        base, ext = os.path.splitext(fn)
        if ext.lower() in {".geoparquet", ".parquet", ".geojson", ".csv"}:
            out.setdefault(base.lower(), {})[ext.lower()] = p
    return out


def resolve_path(layers: Dict[str, Dict[str, str]], keywords: List[str]) -> Optional[str]:
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


def load_any(path: str) -> gpd.GeoDataFrame:
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


def pick_name_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
        if c in gdf.columns:
            return c
    for c in gdf.columns:
        if pd.api.types.is_object_dtype(gdf[c]):
            return c
    return None


def pick_race_count_columns(gdf: gpd.GeoDataFrame) -> Dict[str, Optional[str]]:
    candidates = {
        "total": ["totalE", "TOTAL", "Total", "POP"],
        "white": ["WhiteE", "WHITE", "White"],
        "black": ["BlackE", "BLACK", "Black"],
        "hispanic": ["HispanicE", "HISPANIC", "Hispanic"],
        "asian": ["AsianE", "ASIAN", "Asian"],
        "native": ["NativeE", "NATIVE", "Native"],
        "hipi": ["HIPIE", "NHPIE", "HIPI", "NHPI"],
    }
    out: Dict[str, Optional[str]] = {}
    for k, opts in candidates.items():
        out[k] = next((c for c in opts if c in gdf.columns), None)
    return out


def pick_income_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    cols: List[str] = []
    for c in gdf.columns:
        if pd.api.types.is_numeric_dtype(gdf[c]):
            lc = str(c).lower()
            if ("income" in lc) or ("median" in lc) or c in {"MHI", "MEDHHINC"}:
                cols.append(c)
    return cols[:12]


def pick_weight_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """Pick a tract-level weight column for income weighting.

    Preference: total households, else total population.
    """
    candidates = [
        "Households", "HH", "HHTOTAL", "TotalHouseholds",
        "totalE", "TOTAL", "Total", "POP", "Population", "TOTPOP",
    ]
    for c in candidates:
        if c in gdf.columns and pd.api.types.is_numeric_dtype(gdf[c]):
            return c
    # fall back to any numeric that looks like households/pop
    for c in gdf.columns:
        lc = str(c).lower()
        if ("household" in lc or lc in {"hh", "total", "pop", "population", "totpop"}) and pd.api.types.is_numeric_dtype(gdf[c]):
            return c
    return None


def build_race_by_neighborhood(matches_csv: str, tracts_path: str, name_col: str) -> pd.DataFrame:
    matches = pd.read_csv(matches_csv)
    if matches.empty:
        return pd.DataFrame(columns=[name_col])
    # Determine neighborhood key column in matches
    nb_col = name_col if name_col in matches.columns else (
        "neighborhood" if "neighborhood" in matches.columns else None
    )
    if nb_col is None:
        # Try case-insensitive search
        for c in matches.columns:
            if c.lower() == name_col.lower() or c.lower() == "neighborhood":
                nb_col = c
                break
    if nb_col is None:
        raise SystemExit("Matches CSV lacks a recognizable neighborhood name column.")
    tracts = load_any(tracts_path)
    if tracts is None or tracts.empty:
        return pd.DataFrame(columns=[name_col])

    # Determine tract ID to join
    tract_id = None
    for c in ["GEOID", "GEOID10", "GEOID20", "geoid", "GEOCODE", "ID", "OBJECTID", "tract_id", "TRACT" ]:
        if c in tracts.columns:
            tract_id = c
            break
    if tract_id is None:
        tracts = tracts.reset_index().rename(columns={"index": "tract_id"})
        tract_id = "tract_id"

    cols = pick_race_count_columns(tracts)
    if cols["total"] is None:
        # As a fallback, try percentage columns and compute weighted averages using frac_of_tract
        pcols = [c for c in ["Pwhite", "Pblack", "Phispanic", "Pasian", "Pnative", "PHIPI"] if c in tracts.columns]
        if not pcols:
            return pd.DataFrame(columns=[name_col])
        # Normalize join keys to strings
        matches["tract_id"] = matches["tract_id"].astype(str)
        tr_join = tracts[[tract_id] + pcols].copy()
        tr_join["__tid__"] = tr_join[tract_id].astype(str)
        df = matches.merge(tr_join.drop(columns=[tract_id]), left_on="tract_id", right_on="__tid__", how="left").drop(columns=["__tid__"])
        df = df.dropna(subset=[nb_col])
        w = pd.to_numeric(df.get("frac_of_tract", 0), errors="coerce").fillna(0)
        out = df.groupby(nb_col).apply(lambda g: pd.Series({pc: (pd.to_numeric(g[pc], errors="coerce").fillna(0) * pd.to_numeric(g.get("frac_of_tract", 0), errors="coerce").fillna(0)).sum() / max(pd.to_numeric(g.get("frac_of_tract", 0), errors="coerce").fillna(0).sum(), 1e-9) for pc in pcols})).reset_index()
        # Rename to pct_* convention
        out = out.rename(columns={pc: f"pct_{pc[1:].lower()}" for pc in pcols})
        # Ensure neighborhood column name matches name_col
        if nb_col != name_col and nb_col in out.columns:
            out = out.rename(columns={nb_col: name_col})
        return out

    # Weighted counts using frac_of_tract
    keep = [c for c in cols.values() if c]
    # Normalize join keys to strings
    matches["tract_id"] = matches["tract_id"].astype(str)
    tr_join = tracts[[tract_id] + keep].copy()
    tr_join["__tid__"] = tr_join[tract_id].astype(str)
    df = matches.merge(tr_join.drop(columns=[tract_id]), left_on="tract_id", right_on="__tid__", how="left").drop(columns=["__tid__"])
    df = df.dropna(subset=[nb_col])
    w = pd.to_numeric(df.get("frac_of_tract", 0), errors="coerce").fillna(0)
    # Build weighted sums
    sums = df.groupby(nb_col).apply(lambda g: pd.Series({
        "total_w": (pd.to_numeric(g[cols["total"]], errors="coerce").fillna(0) * pd.to_numeric(g.get("frac_of_tract", 0), errors="coerce").fillna(0)).sum(),
        "white_w": (pd.to_numeric(g.get(cols["white"]), errors="coerce").fillna(0) * pd.to_numeric(g.get("frac_of_tract", 0), errors="coerce").fillna(0)).sum() if cols["white"] else 0.0,
        "black_w": (pd.to_numeric(g.get(cols["black"]), errors="coerce").fillna(0) * pd.to_numeric(g.get("frac_of_tract", 0), errors="coerce").fillna(0)).sum() if cols["black"] else 0.0,
        "hispanic_w": (pd.to_numeric(g.get(cols["hispanic"]), errors="coerce").fillna(0) * pd.to_numeric(g.get("frac_of_tract", 0), errors="coerce").fillna(0)).sum() if cols["hispanic"] else 0.0,
        "asian_w": (pd.to_numeric(g.get(cols["asian"]), errors="coerce").fillna(0) * pd.to_numeric(g.get("frac_of_tract", 0), errors="coerce").fillna(0)).sum() if cols["asian"] else 0.0,
        "native_w": (pd.to_numeric(g.get(cols["native"]), errors="coerce").fillna(0) * pd.to_numeric(g.get("frac_of_tract", 0), errors="coerce").fillna(0)).sum() if cols["native"] else 0.0,
        "hipi_w": (pd.to_numeric(g.get(cols["hipi"]), errors="coerce").fillna(0) * pd.to_numeric(g.get("frac_of_tract", 0), errors="coerce").fillna(0)).sum() if cols["hipi"] else 0.0,
    })).reset_index()
    # Compute percents
    tot = sums["total_w"].replace(0, pd.NA)
    for key, outc in [("white_w", "pct_white"), ("black_w", "pct_black"), ("hispanic_w", "pct_hispanic"), ("asian_w", "pct_asian"), ("native_w", "pct_native"), ("hipi_w", "pct_hipi")]:
        if key in sums.columns:
            sums[outc] = (sums[key] / tot).astype(float)
    # Ensure neighborhood column name matches name_col
    if nb_col != name_col and nb_col in sums.columns:
        sums = sums.rename(columns={nb_col: name_col})
    return sums[[name_col, "pct_white", "pct_black", "pct_hispanic", "pct_asian", "pct_native", "pct_hipi"]].fillna(0.0)


def build_income_by_neighborhood(matches_csv: str, tracts_path: str, name_col: str) -> pd.DataFrame:
    matches = pd.read_csv(matches_csv)
    if matches.empty:
        return pd.DataFrame(columns=[name_col])
    # Determine neighborhood key column in matches
    nb_col = name_col if name_col in matches.columns else (
        "neighborhood" if "neighborhood" in matches.columns else None
    )
    if nb_col is None:
        for c in matches.columns:
            if c.lower() == name_col.lower() or c.lower() == "neighborhood":
                nb_col = c
                break
    if nb_col is None:
        raise SystemExit("Matches CSV lacks a recognizable neighborhood name column.")
    tracts = load_any(tracts_path)
    if tracts is None or tracts.empty:
        return pd.DataFrame(columns=[name_col])

    # Determine tract ID to join
    tract_id = None
    for c in ["GEOID", "GEOID10", "GEOID20", "geoid", "GEOCODE", "ID", "OBJECTID", "tract_id", "TRACT" ]:
        if c in tracts.columns:
            tract_id = c
            break
    if tract_id is None:
        tracts = tracts.reset_index().rename(columns={"index": "tract_id"})
        tract_id = "tract_id"

    income_cols = pick_income_columns(tracts)
    if not income_cols:
        return pd.DataFrame(columns=[name_col])
    # Try to find a weight column (households preferred)
    wcol = pick_weight_column(tracts)
    # Normalize join keys to strings
    matches["tract_id"] = matches["tract_id"].astype(str)
    tr_join = tracts[[tract_id] + income_cols].copy()
    if wcol and wcol in tracts.columns:
        tr_join[wcol] = pd.to_numeric(tracts[wcol], errors="coerce")
    tr_join["__tid__"] = tr_join[tract_id].astype(str)
    df = matches.merge(tr_join.drop(columns=[tract_id]), left_on="tract_id", right_on="__tid__", how="left").drop(columns=["__tid__"])
    df = df.dropna(subset=[nb_col])
    # Weighted average: households/pop weights scaled by fraction-of-tract
    f = pd.to_numeric(df.get("frac_of_tract", 0), errors="coerce").fillna(0)
    if wcol and wcol in df.columns:
        w = pd.to_numeric(df[wcol], errors="coerce").fillna(0) * f
        grouped = df.assign(__w__=w).groupby(nb_col)
        out = grouped.apply(
            lambda g: pd.Series({
                col: (pd.to_numeric(g[col], errors="coerce").fillna(0) * g["__w__"].fillna(0)).sum() / max(g["__w__"].fillna(0).sum(), 1e-9)
                for col in income_cols
            })
        ).reset_index()
    else:
        # Fallback: area-fraction weighting only
        grouped = df.groupby(nb_col)
        out = grouped.apply(
            lambda g: pd.Series({
                col: (pd.to_numeric(g[col], errors="coerce").fillna(0) * pd.to_numeric(g.get("frac_of_tract", 0), errors="coerce").fillna(0)).sum() /
                     max(pd.to_numeric(g.get("frac_of_tract", 0), errors="coerce").fillna(0).sum(), 1e-9)
                for col in income_cols
            })
        ).reset_index()
    # Ensure neighborhood column name matches name_col
    if nb_col != name_col and nb_col in out.columns:
        out = out.rename(columns={nb_col: name_col})
    return out


def main():
    os.makedirs(DERIVED_DIR, exist_ok=True)
    raw_layers = scan_dir(RAW_DIR)
    derived_layers = scan_dir(DERIVED_DIR)
    if not raw_layers:
        log(f"[ERROR] No files in {RAW_DIR}. Run downloads first.")
        sys.exit(1)

    # Resolve neighborhoods geometry
    nbh_path = resolve_path(raw_layers, ["neighborhood"]) or resolve_path(raw_layers, ["neighborhoods"])
    if not nbh_path:
        log("[ERROR] Neighborhoods layer not found in data_raw/")
        sys.exit(1)
    nbh = load_any(nbh_path)
    if nbh is None or nbh.empty:
        log("[ERROR] Neighborhoods failed to load or empty.")
        sys.exit(1)
    name_col = pick_name_col(nbh)
    if not name_col:
        log("[ERROR] Could not detect neighborhood name column.")
        sys.exit(1)

    # Matches CSVs
    race_matches = os.path.join(DERIVED_DIR, "nbh_tract_matches_race.csv")
    income_matches = os.path.join(DERIVED_DIR, "nbh_tract_matches_income.csv")
    if not os.path.exists(race_matches) or not os.path.exists(income_matches):
        log("[ERROR] Matches CSVs not found. Run merge_tracts_to_neighborhoods.py first to create them.")
        sys.exit(1)

    # Resolve tracts
    race_tracts = resolve_path(raw_layers, ["race", "ethnicity", "RVA_Race_and_Ethnicity_2020", "rva_race"]) 
    income_tracts = resolve_path(raw_layers, ["income"]) 
    if not race_tracts or not income_tracts:
        log("[ERROR] Tract source files not found in data_raw/")
        sys.exit(1)

    # Build summaries
    race_by_nbh = build_race_by_neighborhood(race_matches, race_tracts, name_col)
    inc_by_nbh = build_income_by_neighborhood(income_matches, income_tracts, name_col)

    # Save CSVs
    race_csv = os.path.join(DERIVED_DIR, "neighborhoods_from_matches_race.csv")
    inc_csv = os.path.join(DERIVED_DIR, "neighborhoods_from_matches_income.csv")
    race_by_nbh.to_csv(race_csv, index=False)
    inc_by_nbh.to_csv(inc_csv, index=False)
    log(f"[OK] Wrote {race_csv}")
    log(f"[OK] Wrote {inc_csv}")

    # Save GeoParquet with geometry merged
    nbh_wgs = ensure_crs(nbh, 4326)
    try:
        race_pq = os.path.join(DERIVED_DIR, "neighborhoods_from_matches_race.geoparquet")
        inc_pq = os.path.join(DERIVED_DIR, "neighborhoods_from_matches_income.geoparquet")
        nbh_wgs.merge(race_by_nbh, on=name_col, how="left").to_parquet(race_pq, index=False)
        nbh_wgs.merge(inc_by_nbh, on=name_col, how="left").to_parquet(inc_pq, index=False)
        log(f"[OK] Wrote {race_pq}")
        log(f"[OK] Wrote {inc_pq}")
    except Exception as e:
        log(f"[WARN] Failed to write GeoParquet: {e}")

    # Combined (income + race) by neighborhood
    try:
        combined = pd.merge(race_by_nbh, inc_by_nbh, on=name_col, how="outer")
        combined_csv = os.path.join(DERIVED_DIR, "neighborhoods_from_matches_combined.csv")
        combined.to_csv(combined_csv, index=False)
        log(f"[OK] Wrote {combined_csv}")
        combined_pq = os.path.join(DERIVED_DIR, "neighborhoods_from_matches_combined.geoparquet")
        nbh_wgs.merge(combined, on=name_col, how="left").to_parquet(combined_pq, index=False)
        log(f"[OK] Wrote {combined_pq}")
    except Exception as e:
        log(f"[WARN] Failed to write combined neighborhood file: {e}")


if __name__ == "__main__":
    main()
