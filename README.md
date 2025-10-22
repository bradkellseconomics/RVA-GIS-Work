# Richmond GIS Data Downloader & Visualizer

Lightweight scripts to download Richmond GIS datasets, build analysis tables, and produce interactive and printable maps. Everything writes to consistent folders so the workflow is reproducible end-to-end.

## What It Does
- Downloads datasets and stores each one in three formats under `data_raw/`:
  - `<name>.geojson` — canonical raw snapshot (portable, auditable)
  - `<name>.csv` — flat mirror with `lon/lat` (points) or `centroid_lon/centroid_lat` (polygons/lines)
  - `<name>.geoparquet` — fast local cache for analysis
- Builds neighborhood-level metrics combining service-line status and race/ethnicity demographics (with tract?neighborhood areal weighting when polygons are available)
- Generates interactive HTML maps and printable PDFs

## Repo Layout
- `download_all.py` — orchestrator to run all downloads in sequence
- `get_geohub_files.py` — downloads selected ArcGIS Hub datasets
- `serviceline_view_download.py` — downloads the Richmond ServiceLine view
- `race_and_ethnicity.py` — downloads layers from a published Web Map item
- `compute_neighborhood_metrics.py` — creates `data_derived/neighborhoods_enriched.*` from raw layers
- `present_all_maps.py` — generates all interactive HTML maps (+ index page)
- `present_interactive_maps.py` — core interactive maps (neighborhoods, HS zones, service lines, tracts)
- `present_neighborhoods_enriched.py` — choropleths from the enriched neighborhoods file
- `presenting.py` — printable PDF maps of core layers
- `data_raw/` — raw outputs (ignored by Git)
- `data_derived/` — derived/enriched outputs
- `outputs/interactive/` — HTML maps
- `outputs/presenting/` — PDF maps

## Data Workflow (End-to-End)

1) Download raw data (GeoJSON + CSV + GeoParquet in `data_raw/`)
```
python download_all.py
```

2) Build analysis dataset (optional, for neighborhood-level metrics)
```
python compute_neighborhood_metrics.py
```
Writes `data_derived/neighborhoods_enriched.{geojson,geoparquet,csv}` with:
- Lead stats: pct_known_lead, pct_known_nonlead, pct_unknown
- Race/ethnicity: pct_white, pct_black, pct_hispanic, pct_asian, pct_native (and pct_hipi when available)

3) Generate interactive maps (HTML in `outputs/interactive/`)
```
python present_all_maps.py
```
Open `outputs/interactive/index.html` in your browser.

4) Printable PDFs (no basemap)
```
python presenting.py
```
Outputs to `outputs/presenting/`.

## Tract?Neighborhood Weighting
- If a polygon tract layer is present (in `data_raw/` or `rva_layers/`), `compute_neighborhood_metrics.py` uses polygon–polygon overlay in an equal-area CRS (EPSG:5070) and weights tract counts/percents by intersection area. This allows a tract to contribute proportionally to multiple neighborhoods when it spans boundaries.
- If only a CSV (centroids) is available, it falls back to assigning tracts by centroid to neighborhoods and aggregates by counts (or weighted percents if totals exist).

## Quickstart
1) Python environment (3.9+ recommended)

2) Install dependencies (minimal set):
```
pip install geopandas pandas requests pyarrow folium
```

If you hit binary install issues on Windows, consider using conda/mamba:
```
conda install -c conda-forge geopandas pyarrow requests folium
```

3) Run all downloads:
```
python download_all.py
```

Options:
- Only some sources: `python download_all.py --only geohub servicelines`
- Stop on first error: `python download_all.py --stop-on-error`

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
The `.gitignore` excludes large/derived artifacts (`data_raw/`, caches, notebook checkpoints). The repo tracks only code and configuration so it’s safe to publish.

## Troubleshooting
- If race tracts render as points: ensure a polygon tract layer exists (e.g., `RVA_Race_and_Ethnicity_2020.geoparquet`). The presenters and metric builder prefer polygons when available.
- If overlays drop features: the scripts attempt geometry repair (`buffer(0)`). Share filenames if issues persist so we can hard-wire the exact source.
