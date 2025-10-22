import os
import sys
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd


RAW_DIR = "data_raw"
OUT_DIR = os.path.join("data_derived")


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(epsg=epsg)
    return gdf.to_crs(epsg=epsg)


def scan_dir(dir_path: str) -> Dict[str, Dict[str, str]]:
    layers: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(dir_path):
        return layers
    for fn in os.listdir(dir_path):
        p = os.path.join(dir_path, fn)
        if os.path.isfile(p):
            base, ext = os.path.splitext(fn)
            if ext.lower() in {".geoparquet", ".parquet", ".geojson", ".csv"}:
                layers.setdefault(base.lower(), {})[ext.lower()] = p
    return layers


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


def pick_tract_id_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for c in ["GEOID", "GEOID10", "GEOID20", "geoid", "GEOCODE", "OBJECTID", "ID", "NAME"]:
        if c in gdf.columns:
            return c
    return None


def filter_richmond_race_tracts(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Restrict race/ethnicity tracts to Richmond city based on attributes.

    Preference order:
    1) GEOID startswith '51760' (Richmond city FIPS)
    2) NAME contains 'Richmond city'
    3) GEOID startswith '517' (independent cities fallback)
    If none match or masks are empty, returns the original gdf.
    """
    df = gdf.copy()
    masks = []
    # GEOID exact city code first
    for key in ["GEOID", "GEOID10", "GEOID20", "geoid", "GEOCODE", "ID"]:
        if key in df.columns:
            try:
                s = df[key].astype(str)
                m_exact = s.str.startswith("51760")
                if m_exact.any():
                    return df[m_exact].copy()
                masks.append(s.str.startswith("517"))
                break
            except Exception:
                pass
    # NAME contains Richmond city
    if "NAME" in df.columns:
        try:
            masks.append(df["NAME"].astype(str).str.contains("Richmond city", case=False, na=False))
        except Exception:
            pass
    # Fallback combine masks
    if masks:
        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m
        filtered = df[mask].copy()
        if not filtered.empty:
            return filtered
    return df


def pick_race_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    cols: List[str] = []
    for c in ["Pwhite", "Pblack", "Phispanic", "Pasian", "Pnative", "PHIPI"]:
        if c in gdf.columns and pd.api.types.is_numeric_dtype(gdf[c]):
            cols.append(c)
    if not cols:
        for c in gdf.columns:
            if isinstance(c, str) and c.startswith("P") and pd.api.types.is_numeric_dtype(gdf[c]):
                cols.append(c)
    return cols[:12]


def pick_income_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    cols: List[str] = []
    for c in gdf.columns:
        if pd.api.types.is_numeric_dtype(gdf[c]):
            lc = str(c).lower()
            if ("income" in lc) or ("median" in lc) or c in {"MHI", "MEDHHINC"}:
                cols.append(c)
    return cols[:12]


def pick_weight_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    candidates = [
        "Households", "HH", "HHTOTAL", "TotalHouseholds",
        "totalE", "TOTAL", "Total", "POP", "Population", "TOTPOP",
    ]
    for c in candidates:
        if c in gdf.columns and pd.api.types.is_numeric_dtype(gdf[c]):
            return c
    return None


def overlay_weighted_race_counts(nbh: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, name_col: str) -> pd.DataFrame:
    """Use race counts (totalE, WhiteE, BlackE, HispanicE, AsianE, NativeE, HIPIE) to compute neighborhood percentages.
    Tracts are clipped to the union of neighborhood geometries to prevent county spillover.
    """
    nbh_5070 = ensure_crs(nbh, 5070).copy()
    tr_5070 = ensure_crs(tracts, 5070).copy()
    try:
        nbh_5070["geometry"] = nbh_5070.geometry.buffer(0)
        tr_5070["geometry"] = tr_5070.geometry.buffer(0)
    except Exception:
        pass
    # Clip tracts to neighborhood union
    try:
        city_geom = nbh_5070.unary_union
        tr_5070 = gpd.clip(tr_5070, gpd.GeoSeries([city_geom], crs=nbh_5070.crs))
    except Exception:
        pass

    # Map count columns
    count_map = {
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
            if c in tr_5070.columns:
                return c
        return None
    cols = {k: find_col(v) for k, v in count_map.items()}

    # If counts missing, bail to empty
    if cols["total"] is None:
        return pd.DataFrame(columns=[name_col])

    inter = gpd.overlay(nbh_5070[[name_col, "geometry"]], tr_5070, how="intersection", keep_geom_type=False)
    if inter.empty:
        return pd.DataFrame(columns=[name_col])
    tract_area = tr_5070.geometry.area
    right_idx = inter["index_right"].values if "index_right" in inter.columns else inter.index
    w = (inter.geometry.area / tract_area.reindex(right_idx).values).fillna(0)

    def get(cname: Optional[str]) -> pd.Series:
        return pd.to_numeric(inter.get(cname, pd.Series(0, index=inter.index)), errors="coerce").fillna(0)

    dfw = pd.DataFrame({
        name_col: inter[name_col].values,
        "total": (get(cols["total"]) * w).values,
        "white": (get(cols["white"]) * w).values,
        "black": (get(cols["black"]) * w).values,
        "hispanic": (get(cols["hispanic"]) * w).values,
        "asian": (get(cols["asian"]) * w).values,
        "native": (get(cols["native"]) * w).values,
        "hipi": (get(cols["hipi"]) * w).values if cols.get("hipi") else 0,
    })
    agg = dfw.groupby(name_col).sum(numeric_only=True)
    tot = agg["total"]
    cond = tot.gt(0).fillna(False)
    out = pd.DataFrame({name_col: agg.index})
    for k, outc in [("white", "pct_white"), ("black", "pct_black"), ("hispanic", "pct_hispanic"), ("asian", "pct_asian"), ("native", "pct_native"), ("hipi", "pct_hipi")]:
        if k in agg.columns:
            ratio = agg[k] / tot
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            out[outc] = ratio.where(cond, 0.0).fillna(0.0).values
    return out.reset_index(drop=True)


def overlay_weighted_numeric(nbh: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, name_col: str, cols: List[str], weight_col: Optional[str]) -> pd.DataFrame:
    nbh_5070 = ensure_crs(nbh, 5070)
    tr_5070 = ensure_crs(tracts, 5070)
    try:
        nbh_5070 = nbh_5070.copy(); nbh_5070["geometry"] = nbh_5070.geometry.buffer(0)
        tr_5070 = tr_5070.copy(); tr_5070["geometry"] = tr_5070.geometry.buffer(0)
    except Exception:
        pass
    # Keep weights if provided
    keep_cols = ["geometry"] + cols + ([weight_col] if weight_col and weight_col in tr_5070.columns else [])
    tr_slim = tr_5070[keep_cols]
    inter = gpd.overlay(nbh_5070[[name_col, "geometry"]], tr_slim, how="intersection", keep_geom_type=False)
    if inter.empty:
        return pd.DataFrame(columns=[name_col] + cols)
    if weight_col and weight_col in inter.columns:
        w = pd.to_numeric(inter[weight_col], errors="coerce").fillna(0)
        inter["__w__"] = w
    else:
        tract_area = tr_5070.geometry.area
        right_idx = inter["index_right"].values if "index_right" in inter.columns else inter.index
        inter["__w__"] = (inter.geometry.area / tract_area.reindex(right_idx).values).fillna(0)
    dfw = pd.DataFrame({name_col: inter[name_col].values, "__w__": inter["__w__"].values})
    for c in cols:
        if c in inter.columns:
            dfw[c] = pd.to_numeric(inter[c], errors="coerce") * dfw["__w__"]
    agg = dfw.groupby(name_col).sum(numeric_only=True)
    wsum = agg["__w__"]
    cond = wsum.gt(0).fillna(False)
    for c in cols:
        if c in agg.columns:
            ratio = agg[c] / wsum
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            agg[c] = ratio.where(cond, 0.0).fillna(0.0)
    return agg.drop(columns=["__w__"]).reset_index()


def centroid_fill_missing(nbh: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, name_col: str, cols: List[str]) -> pd.DataFrame:
    # Fallback: assign each tract to neighborhood by centroid and average
    nbh_wgs = ensure_crs(nbh, 4326)
    tr_wgs = ensure_crs(tracts, 4326)
    joined = gpd.sjoin(tr_wgs, nbh_wgs[[name_col, "geometry"]], how="left", predicate="within")
    df = pd.DataFrame(joined[[name_col] + [c for c in cols if c in joined.columns]]).copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.groupby(name_col).mean(numeric_only=True).reset_index()


def nearest_fill_missing(nbh: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, name_col: str, cols: List[str]) -> pd.DataFrame:
    # As a last resort, assign each neighborhood centroid the nearest tract values
    nbh_pts = ensure_crs(nbh, 4326).copy()
    nbh_pts["_centroid"] = nbh_pts.geometry.centroid
    nbh_pts = nbh_pts.set_geometry("_centroid")
    tr = ensure_crs(tracts, 4326)
    try:
        nn = gpd.sjoin_nearest(nbh_pts[[name_col, "_centroid"]], tr[["geometry"] + cols], how="left")
    except Exception:
        return pd.DataFrame(columns=[name_col] + cols)
    out = pd.DataFrame(nn[[name_col] + cols]).copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.groupby(name_col).mean(numeric_only=True).reset_index()


def save_matches_report_and_map(nbh: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, name_col: str, tract_label: str):
    """Write a CSV of neighborhood-tract intersections with fractions and an HTML overlay map for QA.

    Outputs:
    - data_derived/nbh_tract_matches_{tract_label}.csv
    - outputs/interactive/nbh_tract_overlay_{tract_label}.html
    """
    # Prepare layers in equal-area CRS, with IDs
    nbh_5070 = ensure_crs(nbh, 5070).copy()
    tr_5070 = ensure_crs(tracts, 5070).copy()
    try:
        nbh_5070["geometry"] = nbh_5070.geometry.buffer(0)
        tr_5070["geometry"] = tr_5070.geometry.buffer(0)
    except Exception:
        pass

    tract_id = pick_tract_id_col(tr_5070)
    if tract_id is None:
        tr_5070["TRACT_ID"] = tr_5070.index.astype(str)
        tract_id = "TRACT_ID"

    # Compute base areas
    nbh_area = nbh_5070.set_index(name_col).geometry.area
    tr_area = tr_5070.set_index(tract_id).geometry.area

    nbh_slim = nbh_5070[[name_col, "geometry"]]
    tr_slim = tr_5070[[tract_id, "geometry"]]

    inter = gpd.overlay(nbh_slim, tr_slim, how="intersection", keep_geom_type=False)
    if inter.empty:
        log(f"[WARN] No intersections found for matches report ({tract_label}).")
        return

    inter_area = inter.geometry.area
    # Map areas
    nbha = nbh_area.reindex(inter[name_col]).values
    tra = tr_area.reindex(inter[tract_id]).values

    df = pd.DataFrame({
        "neighborhood": inter[name_col].values,
        "tract_id": inter[tract_id].values,
        "intersection_area_m2": inter_area.values,
        "neighborhood_area_m2": nbha,
        "tract_area_m2": tra,
    })
    # Fractions (guard divide-by-zero)
    df["frac_of_neighborhood"] = np.where(df["neighborhood_area_m2"] > 0, df["intersection_area_m2"] / df["neighborhood_area_m2"], 0.0)
    df["frac_of_tract"] = np.where(df["tract_area_m2"] > 0, df["intersection_area_m2"] / df["tract_area_m2"], 0.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    matches_csv = os.path.join(OUT_DIR, f"nbh_tract_matches_{tract_label}.csv")
    df.sort_values(["neighborhood", "frac_of_neighborhood"], ascending=[True, False]).to_csv(matches_csv, index=False)
    log(f"[OK] Wrote matches CSV -> {matches_csv}")

    # Build simple overlay map
    try:
        import folium
    except ImportError:
        log("[INFO] folium not installed; skipping overlay map.")
        return

    nbh_wgs = ensure_crs(nbh_slim, 4326)
    tr_wgs = ensure_crs(tr_slim, 4326)
    # Center map on neighborhoods
    try:
        minx, miny, maxx, maxy = nbh_wgs.total_bounds
        lat, lon = (miny + maxy) / 2, (minx + maxx) / 2
    except Exception:
        lat, lon = 37.5407, -77.4360

    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")
    # Neighborhood boundaries (blue)
    folium.GeoJson(
        nbh_wgs.to_json(default=str),
        name="Neighborhoods",
        style_function=lambda _: {"fill": False, "color": "#2b8cbe", "weight": 1.5},
        tooltip=folium.features.GeoJsonTooltip(fields=[name_col], aliases=["Neighborhood"], localize=True),
    ).add_to(m)
    # Tract boundaries (red)
    folium.GeoJson(
        tr_wgs.to_json(default=str),
        name="Census Tracts",
        style_function=lambda _: {"fill": False, "color": "#de2d26", "weight": 1.0},
        tooltip=folium.features.GeoJsonTooltip(fields=[tract_id], aliases=["Tract"], localize=True),
    ).add_to(m)
    folium.LayerControl().add_to(m)

    out_dir = os.path.join("outputs", "interactive")
    os.makedirs(out_dir, exist_ok=True)
    out_html = os.path.join(out_dir, f"nbh_tract_overlay_{tract_label}.html")
    m.save(out_html)
    log(f"[OK] Wrote overlay map -> {out_html}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    layers = scan_dir(RAW_DIR)
    if not layers:
        log(f"[ERROR] No files in {RAW_DIR}. Run downloads first.")
        sys.exit(1)

    nbh_path = resolve_polygon_path(layers, ["neighborhood"])
    if not nbh_path:
        log("[ERROR] Neighborhoods layer not found.")
        sys.exit(1)
    nbh = load_any(nbh_path)
    if nbh is None or nbh.empty:
        log("[ERROR] Neighborhoods failed to load or empty.")
        sys.exit(1)
    name_col = pick_name_col(nbh)
    if not name_col:
        log("[ERROR] Cannot detect neighborhood name column.")
        sys.exit(1)

    # Race tracts
    race_path = resolve_polygon_path(layers, ["race", "ethnicity", "RVA_Race_and_Ethnicity_2020", "rva_race"])
    race_df = None
    if race_path:
        race = load_any(race_path)
        if race is not None and not race.empty:
            # Drop tracts outside Richmond using attributes (NAME/GEOID) before overlay
            race = filter_richmond_race_tracts(race)
            race_df = overlay_weighted_race_counts(nbh, race, name_col)
            # QA: write matches CSV and overlay map
            save_matches_report_and_map(nbh, race, name_col, tract_label="race")
            # Fill missing with centroid assign, then nearest
            missing = set(nbh[name_col]) - set(race_df[name_col])
            if missing:
                pcols = [c for c in ["pct_white", "pct_black", "pct_hispanic", "pct_asian", "pct_native", "pct_hipi"] if c in race_df.columns]
                centroid_df = centroid_fill_missing(nbh, race, name_col, pcols)
                race_df = race_df.merge(centroid_df, on=name_col, how="outer", suffixes=("", "_cent"))
                for c in pcols:
                    if c in race_df.columns and f"{c}_cent" in race_df.columns:
                        race_df[c] = race_df[c].fillna(race_df[f"{c}_cent"]).fillna(0.0)
                        race_df = race_df.drop(columns=[f"{c}_cent"]) 
                missing2 = race_df[name_col].isna().sum() or (len(set(nbh[name_col]) - set(race_df[name_col])))
                if missing2:
                    nearest_df = nearest_fill_missing(nbh, race, name_col, pcols)
                    race_df = race_df.merge(nearest_df, on=name_col, how="outer", suffixes=("", "_near"))
                    for c in pcols:
                        if c in race_df.columns and f"{c}_near" in race_df.columns:
                            race_df[c] = race_df[c].fillna(race_df[f"{c}_near"]).fillna(0.0)
                            race_df = race_df.drop(columns=[f"{c}_near"]) 
        else:
            log("[WARN] Race tracts empty or failed to load.")

    # Income tracts
    income_path = resolve_polygon_path(layers, ["income"])
    income_df = None
    if income_path:
        inc = load_any(income_path)
        if inc is not None and not inc.empty:
            icols = pick_income_columns(inc)
            wcol = pick_weight_column(inc)
            income_df = overlay_weighted_numeric(nbh, inc, name_col, icols, wcol)
            # QA: write matches CSV and overlay map
            save_matches_report_and_map(nbh, inc, name_col, tract_label="income")
            missing = set(nbh[name_col]) - set(income_df[name_col])
            if missing:
                centroid_df = centroid_fill_missing(nbh, inc, name_col, icols)
                income_df = income_df.merge(centroid_df, on=name_col, how="outer", suffixes=("", "_cent"))
                for c in icols:
                    if c in income_df.columns and f"{c}_cent" in income_df.columns:
                        income_df[c] = income_df[c].fillna(income_df[f"{c}_cent"]).fillna(0.0)
                        income_df = income_df.drop(columns=[f"{c}_cent"]) 
                missing2 = income_df[name_col].isna().sum() or (len(set(nbh[name_col]) - set(income_df[name_col])))
                if missing2:
                    nearest_df = nearest_fill_missing(nbh, inc, name_col, icols)
                    income_df = income_df.merge(nearest_df, on=name_col, how="outer", suffixes=("", "_near"))
                    for c in icols:
                        if c in income_df.columns and f"{c}_near" in income_df.columns:
                            income_df[c] = income_df[c].fillna(income_df[f"{c}_near"]).fillna(0.0)
                            income_df = income_df.drop(columns=[f"{c}_near"]) 
        else:
            log("[WARN] Income tracts empty or failed to load.")

    # Skip writing neighborhoods_tract_enriched outputs to avoid confusion.
    # Use build_neighborhood_from_matches.py to compute neighborhood race/income
    # using the intersection fractions produced above.
    log("[INFO] Skipped writing neighborhoods_tract_enriched.*; using matches + builder workflow instead.")


if __name__ == "__main__":
    main()
