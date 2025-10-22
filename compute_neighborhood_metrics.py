import os
import sys
from typing import Dict, Optional, Tuple, List

import geopandas as gpd
import pandas as pd


RAW_DIR = "data_raw"
OUT_DIR = os.path.join("data_derived")


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(epsg=epsg)
    return gdf.to_crs(epsg=epsg)


def _scan_dir(dir_path: str) -> Dict[str, Dict[str, str]]:
    layers: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(dir_path):
        return layers
    for fn in os.listdir(dir_path):
        p = os.path.join(dir_path, fn)
        if os.path.isfile(p):
            base, ext = os.path.splitext(fn)
            ext = ext.lower()
            if ext in {".geojson", ".geoparquet", ".parquet", ".csv"}:
                layers.setdefault(base.lower(), {})[ext] = p
    return layers


def scan_raw() -> Dict[str, Dict[str, str]]:
    # Merge RAW_DIR with optional rva_layers to pick up polygon caches
    layers = _scan_dir(RAW_DIR)
    if os.path.isdir("rva_layers"):
        extra = _scan_dir("rva_layers")
        for base, exts in extra.items():
            if base in layers:
                layers[base].update(exts)
            else:
                layers[base] = dict(exts)
    return layers


def pick_path(layers: Dict[str, Dict[str, str]], keywords: List[str]) -> Optional[str]:
    for k, entry in layers.items():
        if any(kw in k for kw in keywords):
            for ext in (".geoparquet", ".parquet", ".geojson", ".csv"):
                if ext in entry:
                    return entry[ext]
    for entry in layers.values():
        for p in entry.values():
            name = os.path.basename(p).lower()
            if any(kw in name for kw in keywords):
                return p
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


def compute_service_line_stats(points: gpd.GeoDataFrame, neighborhoods: gpd.GeoDataFrame, name_col: str) -> pd.DataFrame:
    pts = points.copy()
    # Derive status_norm from available columns
    src_col = None
    for c in ["bothsidesstatus", "utilmaterial", "custmaterial", "everlead"]:
        if c in pts.columns:
            src_col = c
            break
    if src_col is None:
        pts["status_norm"] = "Unknown"
    else:
        pts["status_norm"] = pts[src_col].map(normalize_status)

    # Spatially join to neighborhoods
    pts_5070 = ensure_crs(pts, 5070)
    nbh_5070 = ensure_crs(neighborhoods, 5070)
    joined = gpd.sjoin(pts_5070, nbh_5070[[name_col, "geometry"]], how="left", predicate="within")

    # Aggregate counts per neighborhood
    grp = joined.groupby(name_col)["status_norm"].value_counts().unstack(fill_value=0)
    for c in ["Lead", "Non-Lead", "Unknown"]:
        if c not in grp.columns:
            grp[c] = 0
    grp["total_points"] = grp[["Lead", "Non-Lead", "Unknown"]].sum(axis=1)
    grp["known_points"] = grp[["Lead", "Non-Lead"]].sum(axis=1)
    # Rates
    grp["pct_known_lead"] = (grp["Lead"] / grp["known_points"]).where(grp["known_points"] > 0, 0)
    grp["pct_known_nonlead"] = (grp["Non-Lead"] / grp["known_points"]).where(grp["known_points"] > 0, 0)
    grp["pct_unknown"] = (grp["Unknown"] / grp["total_points"]).where(grp["total_points"] > 0, 0)
    # Also compute rates out of total points
    grp["pct_known_lead_total"] = (grp["Lead"] / grp["total_points"]).where(grp["total_points"] > 0, 0)
    grp["pct_known_nonlead_total"] = (grp["Non-Lead"] / grp["total_points"]).where(grp["total_points"] > 0, 0)
    return grp.reset_index()


def pick_name_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neigh_name", "neighborhood_name"]:
        if c in gdf.columns:
            return c
    # fallback: first object dtype column
    for c in gdf.columns:
        if pd.api.types.is_object_dtype(gdf[c]):
            return c
    return None


def compute_race_weighted(neighborhoods: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, name_col: str) -> pd.DataFrame:
    def is_polygonal(g: gpd.GeoDataFrame) -> bool:
        try:
            types = g.geom_type.dropna().unique().tolist() if hasattr(g, "geom_type") else []
            return any("Polygon" in t for t in types)
        except Exception:
            return False

    # Identify tract id column to carry through overlay
    tract_id_col = None
    for c in ["GEOID", "GEOID10", "GEOID20", "geoid", "GEOCODE", "OBJECTID", "ID"]:
        if c in tracts.columns:
            tract_id_col = c
            break
    if tract_id_col is None:
        tracts = tracts.copy()
        tract_id_col = "__tract_id__"
        tracts[tract_id_col] = tracts.index.astype(str)

    # Choose count columns if available
    count_cols = {
        "total": ["totalE", "TOTAL", "Total", "POP"],
        "white": ["WhiteE", "WHITE", "White"],
        "black": ["BlackE", "BLACK", "Black"],
        "hispanic": ["HispanicE", "HISPANIC", "Hispanic"],
        "asian": ["AsianE", "ASIAN", "Asian"],
        "native": ["NativeE", "NATIVE", "Native"],
        "hipi": ["HIPIE", "NHPIE", "HIPI", "NHPI"],
    }

    def find_col(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in tracts.columns:
                return c
        return None

    cols = {k: find_col(v) for k, v in count_cols.items()}
    use_percent = cols["total"] is None

    # Also look for percent columns
    pcols = {
        "pct_white": "Pwhite" if "Pwhite" in tracts.columns else None,
        "pct_black": "Pblack" if "Pblack" in tracts.columns else None,
        "pct_hispanic": "Phispanic" if "Phispanic" in tracts.columns else None,
        "pct_asian": "Pasian" if "Pasian" in tracts.columns else ("AsianP" if "AsianP" in tracts.columns else None),
        "pct_native": "Pnative" if "Pnative" in tracts.columns else None,
        "pct_hipi": "PHIPI" if "PHIPI" in tracts.columns else None,
    }
    # Case 1: polygonal tracts available -> areal weighting via overlay
    if is_polygonal(tracts):
        # Project and repair geometries in equal-area CRS
        nbh_5070 = ensure_crs(neighborhoods, 5070)
        tr_5070 = ensure_crs(tracts, 5070)
        try:
            nbh_5070 = nbh_5070.copy()
            nbh_5070["geometry"] = nbh_5070.geometry.buffer(0)
        except Exception:
            pass
        try:
            tr_5070 = tr_5070.copy()
            tr_5070["geometry"] = tr_5070.geometry.buffer(0)
        except Exception:
            pass

        # Keep only necessary columns from tracts for overlay
        tract_keep_cols = [tract_id_col, "geometry"]
        for c in list(cols.values()) + list(pcols.values()):
            if c and c in tr_5070.columns:
                tract_keep_cols.append(c)
        tr_keep = tr_5070[tract_keep_cols].copy()

        inter = gpd.overlay(nbh_5070[[name_col, "geometry"]], tr_keep, how="intersection", keep_geom_type=False)
        if inter.empty:
            log("[WARN] Neighborhood/tract overlay is empty after intersection. Falling back to centroid assignment.")
            # Fallthrough to centroid approach below
        else:
            tract_area = tr_5070.set_index(tract_id_col).geometry.area
            inter_area = inter.geometry.area
            inter = inter.assign(__w__=(inter_area / tract_area.reindex(inter[tract_id_col]).values))
            inter["__w__"] = inter["__w__"].fillna(0).clip(lower=0)

            if not use_percent and cols["total"] is not None:
                def get(series_name: Optional[str]) -> pd.Series:
                    return pd.to_numeric(inter.get(series_name, pd.Series(0, index=inter.index)), errors="coerce").fillna(0)

                dfw = pd.DataFrame(
                    {
                        name_col: inter[name_col].values,
                        "total_count": (get(cols["total"]) * inter["__w__"]).values,
                        "white_count": (get(cols["white"]) * inter["__w__"]).values,
                        "black_count": (get(cols["black"]) * inter["__w__"]).values,
                        "hispanic_count": (get(cols["hispanic"]) * inter["__w__"]).values,
                        "asian_count": (get(cols["asian"]) * inter["__w__"]).values,
                        "native_count": (get(cols["native"]) * inter["__w__"]).values,
                        "hipi_count": (get(cols["hipi"]) * inter["__w__"]).values if cols.get("hipi") else 0,
                    }
                )
                agg = dfw.groupby(name_col).sum()
                total = agg["total_count"].replace(0, pd.NA)
                agg["pct_white"] = (agg["white_count"] / total).astype(float)
                agg["pct_black"] = (agg["black_count"] / total).astype(float)
                agg["pct_hispanic"] = (agg["hispanic_count"] / total).astype(float)
                agg["pct_asian"] = (agg["asian_count"] / total).astype(float)
                agg["pct_native"] = (agg["native_count"] / total).astype(float)
                if "hipi_count" in agg.columns:
                    agg["pct_hipi"] = (agg["hipi_count"] / total).astype(float)
                return agg.reset_index()
            else:
                dfw = pd.DataFrame({name_col: inter[name_col].values, "__w__": inter["__w__"].values})
                for out_col, src_col in pcols.items():
                    if src_col and src_col in inter.columns:
                        dfw[out_col] = pd.to_numeric(inter[src_col], errors="coerce") * dfw["__w__"]
                agg = dfw.groupby(name_col).sum()
                wsum = agg["__w__"].replace(0, pd.NA)
                for col in [c for c in agg.columns if c.startswith("pct_")]:
                    agg[col] = (agg[col] / wsum).astype(float)
                return agg.drop(columns=["__w__"]).reset_index()

    # Case 2: no polygon tracts — centroid assignment fallback
    nbh = ensure_crs(neighborhoods, 4326)
    tr = ensure_crs(tracts, 4326)
    try:
        joined = gpd.sjoin(tr, nbh[[name_col, "geometry"]], how="left", predicate="within")
    except Exception:
        joined = gpd.sjoin(tr, nbh[[name_col, "geometry"]], how="left")

    # If counts available, aggregate counts; else area-unavailable, weight percentages by total if available, else unweighted mean
    if not use_percent and cols["total"] is not None:
        def get(series_name: Optional[str]) -> pd.Series:
            return pd.to_numeric(joined.get(series_name, pd.Series(0, index=joined.index)), errors="coerce").fillna(0)

        dfw = pd.DataFrame(
            {
                name_col: joined[name_col].values,
                "total_count": get(cols["total"]).values,
                "white_count": get(cols["white"]).values,
                "black_count": get(cols["black"]).values,
                "hispanic_count": get(cols["hispanic"]).values,
                "asian_count": get(cols["asian"]).values,
                "native_count": get(cols["native"]).values,
                "hipi_count": get(cols["hipi"]).values if cols.get("hipi") else 0,
            }
        )
        agg = dfw.groupby(name_col).sum()
        total = agg["total_count"].replace(0, pd.NA)
        agg["pct_white"] = (agg["white_count"] / total).astype(float)
        agg["pct_black"] = (agg["black_count"] / total).astype(float)
        agg["pct_hispanic"] = (agg["hispanic_count"] / total).astype(float)
        agg["pct_asian"] = (agg["asian_count"] / total).astype(float)
        agg["pct_native"] = (agg["native_count"] / total).astype(float)
        if "hipi_count" in agg.columns:
            agg["pct_hipi"] = (agg["hipi_count"] / total).astype(float)
        return agg.reset_index()
    else:
        # Weight percent columns by total if available; else simple mean
        weights = pd.to_numeric(joined.get(cols["total"], pd.Series(1, index=joined.index)), errors="coerce").fillna(1)
        dfw = pd.DataFrame({name_col: joined[name_col].values, "__w__": weights.values})
        for out_col, src_col in pcols.items():
            if src_col and src_col in joined.columns:
                dfw[out_col] = pd.to_numeric(joined[src_col], errors="coerce") * dfw["__w__"]
        agg = dfw.groupby(name_col).sum()
        wsum = agg["__w__"].replace(0, pd.NA)
        for col in [c for c in agg.columns if c.startswith("pct_")]:
            agg[col] = (agg[col] / wsum).astype(float)
        return agg.drop(columns=["__w__"]).reset_index()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    layers = scan_raw()
    if not layers:
        log(f"[ERROR] No files in {RAW_DIR}. Run downloads first.")
        sys.exit(1)

    # Load neighborhoods
    nbh_path = pick_path(layers, ["neighborhood"])
    if not nbh_path:
        log("[ERROR] Neighborhoods layer not found in data_raw.")
        sys.exit(1)
    neighborhoods = load_any(nbh_path)
    name_col = pick_name_col(neighborhoods)
    if name_col is None:
        log("[ERROR] Could not determine neighborhood name column.")
        sys.exit(1)

    # Service lines
    sl_path = pick_path(layers, ["serviceline", "service_line", "richmond_serviceline_view"]) 
    if not sl_path:
        log("[WARN] Service lines dataset not found; lead metrics will be missing.")
        sl_stats = pd.DataFrame(columns=[name_col])
    else:
        sl = load_any(sl_path)
        if sl is None or sl.empty:
            log("[WARN] Service lines dataset empty; lead metrics will be missing.")
            sl_stats = pd.DataFrame(columns=[name_col])
        else:
            sl_stats = compute_service_line_stats(sl, neighborhoods, name_col)

    # Race/Ethnicity tracts
    race_path = None
    tracts = None
    polygon_candidate = None
    polygon_gdf = None
    # Prefer polygonal source if available; otherwise accept any table with the right columns
    for entry in layers.values():
        for p in entry.values():
            g = load_any(p)
            if g is None:
                continue
            has_cols = any(c in g.columns for c in ["totalE", "WhiteE", "BlackE", "HispanicE", "AsianE", "NativeE", "Pblack", "Pwhite", "Phispanic"])
            if not has_cols:
                continue
            try:
                types = g.geom_type.dropna().unique().tolist() if hasattr(g, "geom_type") else []
            except Exception:
                types = []
            if any("Polygon" in t for t in types):
                polygon_candidate = p
                polygon_gdf = g
                break
            if race_path is None:
                race_path = p
                tracts = g
        if polygon_candidate:
            break
    if polygon_candidate:
        race_path = polygon_candidate
        tracts = polygon_gdf
    if not race_path:
        log("[WARN] Race/Ethnicity tracts not found; demographic metrics will be missing.")
        race_stats = pd.DataFrame(columns=[name_col])
    else:
        log(f"[INFO] Using race source: {os.path.basename(race_path)} | rows={len(tracts)}")
        race_stats = compute_race_weighted(neighborhoods, tracts, name_col)

    # Merge onto neighborhoods geometry
    nbh_wgs84 = ensure_crs(neighborhoods, 4326)
    out = nbh_wgs84.merge(sl_stats, on=name_col, how="left").merge(race_stats, on=name_col, how="left")

    # Simple coverage report
    try:
        covered = out["pct_black"].notna().sum() if "pct_black" in out.columns else out["pct_white"].notna().sum() if "pct_white" in out.columns else 0
        log(f"[INFO] Neighborhoods with race metrics: {covered} / {len(out)}")
    except Exception:
        pass

    # Save triad
    base = os.path.join(OUT_DIR, "neighborhoods_enriched")
    geojson_path = base + ".geojson"
    csv_path = base + ".csv"
    pq_path = base + ".geoparquet"

    try:
        out.to_file(geojson_path, driver="GeoJSON")
    except Exception as e:
        log(f"[WARN] Failed to write GeoJSON: {e}")
    try:
        out.to_parquet(pq_path, index=False)
    except Exception as e:
        log(f"[WARN] Failed to write GeoParquet: {e}")
    try:
        df = out.copy()
        if "geometry" in df.columns:
            df = df.drop(columns=["geometry"])  # type: ignore[arg-type]
        df.to_csv(csv_path, index=False)
    except Exception as e:
        log(f"[WARN] Failed to write CSV: {e}")

    log(f"[OK] Wrote outputs -> {geojson_path}, {pq_path}, {csv_path}")


if __name__ == "__main__":
    main()
