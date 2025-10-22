import os
import sys
from typing import Optional, Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt


DATA_DIR = "data_raw"
OUT_DIR = os.path.join("outputs", "presenting")


def _ensure_crs_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        if gdf.crs is None:
            return gdf.set_crs(epsg=4326)
        return gdf.to_crs(epsg=4326)
    except Exception:
        return gdf


def _read_any(path: str) -> Optional[gpd.GeoDataFrame]:
    try:
        if path.lower().endswith(('.geoparquet', '.parquet')):
            return gpd.read_parquet(path)
        if path.lower().endswith('.geojson'):
            return gpd.read_file(path)
        # Fallback: try geofile read
        return gpd.read_file(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}", file=sys.stderr)
        return None


def _scan_data_raw() -> Dict[str, Dict[str, str]]:
    files: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(DATA_DIR):
        return files
    for fn in os.listdir(DATA_DIR):
        p = os.path.join(DATA_DIR, fn)
        if not os.path.isfile(p):
            continue
        base, ext = os.path.splitext(fn)
        if ext.lower() in {'.geojson', '.geoparquet', '.parquet'}:
            files.setdefault(base.lower(), {})[ext.lower()] = p
    return files


def _resolve_layer(layers: Dict[str, Dict[str, str]], keywords: List[str]) -> Optional[str]:
    # Prefer exact base name match, otherwise substring match
    for k in layers:
        if any(k.find(w) >= 0 for w in keywords):
            # prefer geoparquet > geojson > parquet
            entry = layers[k]
            for ext in ('.geoparquet', '.geojson', '.parquet'):
                if ext in entry:
                    return entry[ext]
    # fallback: search filenames
    for entry in layers.values():
        for path in entry.values():
            name = os.path.basename(path).lower()
            if any(w in name for w in keywords):
                return path
    return None


def _common_extent(gdfs: List[gpd.GeoDataFrame]) -> Optional[Tuple[float, float, float, float]]:
    bounds = []
    for g in gdfs:
        if g is not None and hasattr(g, 'total_bounds') and len(g) > 0 and g.geometry.notna().any():
            try:
                b = _ensure_crs_wgs84(g).total_bounds
                bounds.append(b)
            except Exception:
                pass
    if not bounds:
        return None
    import numpy as np
    arr = np.array(bounds)
    return (arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max())


def _save_pdf(fig, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


def _plot_polygons(gdf: gpd.GeoDataFrame, title: str, out_path: str, color_by: Optional[str] = None):
    gdf = _ensure_crs_wgs84(gdf)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    try:
        if color_by and color_by in gdf.columns:
            scheme_kwargs = {}
            try:
                import mapclassify  # noqa: F401
                scheme_kwargs = dict(scheme='Quantiles', k=5)
            except Exception:
                scheme_kwargs = {}
            gdf.plot(ax=ax, column=color_by, cmap='OrRd', edgecolor='#666', linewidth=0.4,
                     legend=True, **scheme_kwargs)
        else:
            gdf.plot(ax=ax, facecolor='#b3cde3', edgecolor='#6497b1', linewidth=0.5, alpha=0.8)
    except Exception as e:
        print(f"[WARN] Polygon plot failed: {e}", file=sys.stderr)
        gdf.boundary.plot(ax=ax, color='#666', linewidth=0.6)

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    _save_pdf(fig, out_path)


def _plot_points(gdf: gpd.GeoDataFrame, title: str, out_path: str, sample: int = 40000):
    gdf = _ensure_crs_wgs84(gdf)
    if len(gdf) > sample:
        gdf = gdf.sample(sample, random_state=1)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    try:
        gdf.plot(ax=ax, markersize=3, color='#1f78b4', alpha=0.6, linewidth=0)
    except Exception as e:
        print(f"[WARN] Point plot failed: {e}", file=sys.stderr)
        # Attempt to derive lon/lat
        if {'lon', 'lat'}.issubset(gdf.columns):
            import pandas as pd
            from shapely.geometry import Point
            gdf = gpd.GeoDataFrame(gdf, geometry=[Point(xy) for xy in zip(gdf['lon'], gdf['lat'])], crs='EPSG:4326')
            gdf.plot(ax=ax, markersize=3, color='#1f78b4', alpha=0.6, linewidth=0)
        else:
            raise
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    _save_pdf(fig, out_path)


def main():
    layers = _scan_data_raw()
    if not layers:
        print(f"[ERROR] No files found in {DATA_DIR}. Run downloads first.", file=sys.stderr)
        sys.exit(1)

    # Resolve paths
    neighborhoods_path = _resolve_layer(layers, ["neighborhood"])
    hs_zones_path = _resolve_layer(layers, ["high_school_zones", "highschool", "school_zone", "zone"])
    servicelines_path = _resolve_layer(layers, ["serviceline", "richmond_serviceline_view", "service_line"])

    race_path = None
    # Try to find a layer with Pblack/Phispanic columns
    for entry in layers.values():
        for p in entry.values():
            g = _read_any(p)
            if g is not None and any(c in g.columns for c in ["Pblack", "Phispanic", "Pwhite", "Pnwhite"]):
                race_path = p
                break
        if race_path:
            break

    # Load
    nbh = _read_any(neighborhoods_path) if neighborhoods_path else None
    hs = _read_any(hs_zones_path) if hs_zones_path else None
    sl = _read_any(servicelines_path) if servicelines_path else None
    race = _read_any(race_path) if race_path else None

    # Plot PDFs
    if nbh is not None and not nbh.empty:
        _plot_polygons(nbh, title="Neighborhoods", out_path=os.path.join(OUT_DIR, "neighborhoods.pdf"))
    else:
        print("[WARN] Neighborhoods layer not found.")

    if hs is not None and not hs.empty:
        _plot_polygons(hs, title="High School Zones", out_path=os.path.join(OUT_DIR, "high_school_zones.pdf"))
    else:
        print("[WARN] High School Zones layer not found.")

    if sl is not None and not sl.empty:
        _plot_points(sl, title="Service Lines (sample if large)", out_path=os.path.join(OUT_DIR, "service_lines.pdf"))
    else:
        print("[WARN] Service Lines layer not found.")

    if race is not None and not race.empty:
        # Prefer Pblack if present
        col = "Pblack" if "Pblack" in race.columns else None
        _plot_polygons(race, title=f"Race/Ethnicity{(' – ' + col) if col else ''}", out_path=os.path.join(OUT_DIR, "race_ethnicity_tracts.pdf"), color_by=col)
    else:
        print("[WARN] Race/Ethnicity layer not found.")


if __name__ == "__main__":
    main()

