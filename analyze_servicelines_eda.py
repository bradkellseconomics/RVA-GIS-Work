import os
import sys
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd


ANALYSIS_DIR = os.path.join("data", "analysis")


def log(msg: str) -> None:
    print(msg, flush=True)


KEEP_COLS = [
    "utilmaterial",
    "custmaterial",
    "bothsidesstatus",
    "yearstructbuilt",
    "buildingtype",
    "MHI",
    "Pblack",
    "Pwhite",
    "Phispanic",
    "utilverified",
    "custverified",
    "J40_TotCatExceeded",
    "neighborhood_name",
    "lat",
    "lon",
    "LCRRSampleTier",
    "ServiceType",
    "sensitivepop",
    "disadvantaged",
]


def load_master() -> gpd.GeoDataFrame:
    base = os.path.join(ANALYSIS_DIR, "servicelines_house_with_attributes")
    for ext in (".geoparquet", ".parquet", ".geojson", ".csv"):
        p = base + ext
        if os.path.exists(p):
            try:
                if ext in (".geoparquet", ".parquet"):
                    return gpd.read_parquet(p)
                if ext == ".geojson":
                    return gpd.read_file(p)
                if ext == ".csv":
                    df = pd.read_csv(p, low_memory=False)
                    # Build geometry if lon/lat exist
                    if {"lon", "lat"}.issubset(df.columns):
                        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
                    return gpd.GeoDataFrame(df, geometry=None)
            except Exception as e:
                log(f"[WARN] Failed to read {p}: {e}")
    raise SystemExit("Master file servicelines_house_with_attributes not found in data/analysis/")


def normalize_status(val: Optional[str]) -> str:
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


def build_minimal(df: gpd.GeoDataFrame) -> pd.DataFrame:
    cols = [c for c in KEEP_COLS if c in df.columns]
    out = pd.DataFrame(df[cols].copy())
    # Create normalized categories
    if "utilmaterial" in out.columns:
        out["utilmaterial_cat"] = out["utilmaterial"].map(normalize_status)
    if "custmaterial" in out.columns:
        out["custmaterial_cat"] = out["custmaterial"].map(normalize_status)
    if "bothsidesstatus" in out.columns:
        out["bothsidesstatus_cat"] = out["bothsidesstatus"].map(normalize_status)
    # Clean yearstructbuilt into numeric
    if "yearstructbuilt" in out.columns:
        y = pd.to_numeric(out["yearstructbuilt"], errors="coerce")
        current_year = pd.Timestamp.now().year
        # Clean zeros and future years only; keep true pre-1800 structures as-is
        valid = (y > 0) & (y <= current_year)
        out.loc[~valid, "yearstructbuilt"] = pd.NA
    return out


def summarize(min_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Identify numeric (continuous) vs categorical
    numeric_cols: List[str] = []
    cat_cols: List[str] = []
    for c in min_df.columns:
        if c in {"lat", "lon"} or pd.api.types.is_numeric_dtype(min_df[c]):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)
    # Always treat these cats as categorical
    for c in ["utilmaterial", "custmaterial", "bothsidesstatus", "utilmaterial_cat", "custmaterial_cat", "bothsidesstatus_cat", "buildingtype", "utilverified", "custverified", "LCRRSampleTier", "ServiceType", "sensitivepop", "disadvantaged", "neighborhood_name"]:
        if c in min_df.columns:
            if c in numeric_cols:
                numeric_cols.remove(c)
            if c not in cat_cols:
                cat_cols.append(c)

    # Numeric summary
    num_summary_rows: List[Dict[str, object]] = []
    for c in numeric_cols:
        s = pd.to_numeric(min_df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        num_summary_rows.append({
            "variable": c,
            "count": int(s.notna().sum()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "min": float(s.min()),
            "p10": float(s.quantile(0.10)),
            "p90": float(s.quantile(0.90)),
            "max": float(s.max()),
        })
    num_summary = pd.DataFrame(num_summary_rows)

    # Categorical summary
    cat_summary_rows: List[Dict[str, object]] = []
    n_rows = len(min_df)
    for c in cat_cols:
        counts = min_df[c].astype("string").fillna("<NA>").value_counts(dropna=False)
        for k, v in counts.items():
            cat_summary_rows.append({
                "variable": c,
                "category": k,
                "count": int(v),
                "percent": float(v) / n_rows if n_rows else 0.0,
            })
    cat_summary = pd.DataFrame(cat_summary_rows)

    return {"numeric": num_summary, "categorical": cat_summary}


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    gdf = load_master()
    min_df = build_minimal(gdf)
    out_csv = os.path.join(ANALYSIS_DIR, "servicelines_eda_minimal.csv")
    min_df.to_csv(out_csv, index=False)
    log(f"[OK] Wrote {out_csv}")

    stats = summarize(min_df)
    num_csv = os.path.join(ANALYSIS_DIR, "servicelines_eda_summary_numeric.csv")
    cat_csv = os.path.join(ANALYSIS_DIR, "servicelines_eda_summary_categorical.csv")
    stats["numeric"].to_csv(num_csv, index=False)
    stats["categorical"].to_csv(cat_csv, index=False)
    log(f"[OK] Wrote {num_csv}")
    log(f"[OK] Wrote {cat_csv}")


if __name__ == "__main__":
    main()
