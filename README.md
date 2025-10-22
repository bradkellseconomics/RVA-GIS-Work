# Richmond GIS Data Downloader

Lightweight scripts to download Richmond GIS datasets (GeoHub layers, service line view, and race/ethnicity layers) into a consistent raw data folder for analysis.

## What It Does
- Fetches datasets and stores each one in three formats under `data_raw/`:
  - `<name>.geojson` — canonical raw snapshot (portable, auditable)
  - `<name>.csv` — flat mirror with `lon/lat` (points) or `centroid_lon/centroid_lat` (polygons/lines)
  - `<name>.geoparquet` — fast local cache for analysis

## Repo Layout
- `download_all.py` — orchestrator to run all downloads in sequence
- `get_geohub_files.py` — downloads selected ArcGIS Hub datasets
- `serviceline_view_download.py` — downloads the Richmond ServiceLine view
- `race_and_ethnicity.py` — downloads layers from a published Web Map item
- `data_raw/` — raw outputs (ignored by Git); contains GeoJSON/CSV/GeoParquet

## Quickstart
1) Python environment (3.9+ recommended)

2) Install dependencies (minimal set):
```
pip install geopandas pandas requests pyarrow
```

If you hit binary install issues on Windows, consider using conda/mamba:
```
conda install -c conda-forge geopandas pyarrow requests
```

3) Run all downloads:
```
python download_all.py
```

Options:
- Only some sources:
```
python download_all.py --only geohub servicelines
```
- Stop on first error:
```
python download_all.py --stop-on-error
```

## Data Output
All files are written to `data_raw/` with sanitized, lowercase names (alphanumeric/underscore).

Examples:
- `data_raw/high_school_zones.geojson`
- `data_raw/high_school_zones.csv`
- `data_raw/high_school_zones.geoparquet`

## Environment / Secrets
No API keys are required for the current data sources. If you need HTTP(S) proxies, set standard environment variables before running:
```
set HTTP_PROXY=http://proxy:8080
set HTTPS_PROXY=http://proxy:8080
```

An example `.env.example` is included for convenience. These scripts do not read `.env` directly; use your shell env or adapt the scripts if you prefer `python-dotenv`.

## Version Control
The `.gitignore` excludes large/derived artifacts (`data_raw/`, caches, notebooks’ checkpoints). The repo tracks only code and configuration so it’s safe to publish.

