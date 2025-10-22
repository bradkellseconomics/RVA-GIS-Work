import re
from pathlib import Path

import geopandas as gpd
import requests


# Define datasets as name: URL pairs
DATASETS = {
    "high_school_zones": "https://richmond-geo-hub-cor.hub.arcgis.com/datasets/7207a12e4c5d47bc9bd4233f81e7adbc_0",
    "neighborhoods": "https://richmond-geo-hub-cor.hub.arcgis.com/datasets/7a0ffef23d16461e9728c065f27b2790_0",
}

# Unified raw output directory (GeoJSON + CSV mirrors)
RAW_OUTDIR = Path("data_raw")
RAW_OUTDIR.mkdir(exist_ok=True)


def get_dataset_id(url: str) -> str:
    """Extract the ArcGIS Hub dataset ID from any dataset URL."""
    m = re.search(r"([0-9a-f]{32}_[0-9]+)", url)
    if not m:
        raise ValueError(f"Cannot extract dataset ID from: {url}")
    return m.group(1)


def get_geojson_url(dataset_id: str) -> str:
    """Try official ArcGIS Hub API first; fall back to predictable pattern."""
    meta_url = f"https://opendata.arcgis.com/api/v3/datasets/{dataset_id}"
    meta = requests.get(meta_url, timeout=30).json()

    # Prefer listed download link
    for dl in meta.get("downloads", []):
        if dl.get("format", "").lower() == "geojson":
            return dl["url"]

    # Fallback: construct predictable pattern
    return (
        f"https://opendata.arcgis.com/api/v3/datasets/{dataset_id}/downloads/data?format=geojson&spatialRefId=4326"
    )


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_")


def _to_csv_with_coords(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = gdf.copy()
    try:
        if df.crs is None:
            df = df.set_crs(epsg=4326)
    except Exception:
        pass
    try:
        if hasattr(df, "geometry") and "geometry" in df.columns and df.geometry.notna().any():
            geom_types = set(df.geometry.geom_type.dropna().unique().tolist())
            if geom_types.issubset({"Point", "MultiPoint"}):
                pts = df.to_crs(epsg=4326)
                df["lon"] = pts.geometry.x
                df["lat"] = pts.geometry.y
            else:
                proj = df.to_crs(epsg=3857)
                cent = proj.geometry.centroid.to_crs(epsg=4326)
                df["centroid_lon"] = cent.x
                df["centroid_lat"] = cent.y
    except Exception:
        # Leave as-is if we cannot compute coordinates
        pass
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])  # type: ignore[arg-type]
    return df


def download_dataset(name: str, dataset_url: str):
    dataset_id = get_dataset_id(dataset_url)
    print(f"[INFO] {name}: {dataset_id}")

    geojson_url = get_geojson_url(dataset_id)
    print(f"[INFO] Downloading GeoJSON from {geojson_url}")

    gdf = gpd.read_file(geojson_url)
    base = _sanitize(name)
    geojson_path = RAW_OUTDIR / f"{base}.geojson"
    csv_path = RAW_OUTDIR / f"{base}.csv"

    # Save raw GeoJSON
    gdf.to_file(geojson_path, driver="GeoJSON")

    # Save CSV mirror
    gdf_csv = _to_csv_with_coords(gdf)
    gdf_csv.to_csv(csv_path, index=False)

    # Save GeoParquet cache alongside raw files
    geopq_path = RAW_OUTDIR / f"{base}.geoparquet"
    try:
        gdf.to_parquet(geopq_path, index=False)
    except Exception as e:
        print(f"[WARN] Failed to write GeoParquet for {name}: {e}")

    print(f"[OK] Saved {geojson_path}, {csv_path}, and {geopq_path} ({len(gdf)} features)")


def main():
    for name, url in DATASETS.items():
        try:
            download_dataset(name, url)
        except Exception as e:
            print(f"[ERROR] Failed for {name}: {e}")


if __name__ == "__main__":
    main()
