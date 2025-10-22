# Richmond GIS: Downloads, Aggregation, and Maps

Lightweight scripts to download Richmond GIS datasets, aggregate tract data to neighborhoods, build a house‑level master file, and produce interactive and printable maps. The workflow is modular and reproducible.

## Data Layout
- `data_raw/` — raw inputs (GeoJSON/CSV/GeoParquet) written by download scripts
- `data_derived/` — derived outputs (matches, neighborhood summaries)
- `data/analysis/` — analysis-friendly CSV/Parquet (master file, stats)
- `outputs/interactive/` — HTML maps
- `outputs/presenting/` — PDF maps

## Scripts
- Downloads
  - `download_all.py` — orchestrator (GeoHub layers, service lines, race/ethnicity, income)
  - `get_geohub_files.py` — selected ArcGIS Hub datasets (e.g., neighborhoods, HS zones)
  - `serviceline_view_download.py` — Richmond ServiceLine view (houses)
  - `race_and_ethnicity.py` — race/ethnicity layers (tracts)
  - `download_income.py` — income layers (tracts)
- Tract→Neighborhood aggregation (recommended path)
  - `merge_tracts_to_neighborhoods.py` — builds tract↔neighborhood matches via overlay, writes:
    - `data_derived/nbh_tract_matches_race.csv`
    - `data_derived/nbh_tract_matches_income.csv`
    - QA maps: `outputs/interactive/nbh_tract_overlay_{race|income}.html`
  - `build_neighborhood_from_matches.py` — computes neighborhood race (% shares) and income (weighted averages) using the match fractions, writes:
    - `data_derived/neighborhoods_from_matches_race.{csv,geoparquet}`
    - `data_derived/neighborhoods_from_matches_income.{csv,geoparquet}`
- House‑level master
  - `master_file_creation.py` — joins service lines to neighborhoods, and directly enriches houses with tract‑level race and income (polygon joins), writes:
    - `data/analysis/servicelines_house_with_attributes.{csv,geoparquet}`
- Maps
  - `present_all_maps.py` — generates all interactive maps (tract race/income, neighborhood from matches, core layers) and index page
  - `presenting.py` — printable PDFs for core layers

## Workflow
1) Download raw data
   - `python download_all.py`
2) Build tract↔neighborhood matches + QA
   - `python merge_tracts_to_neighborhoods.py`
   - Inspect: `data_derived/nbh_tract_matches_{race|income}.csv` and overlay HTML in `outputs/interactive/`
3) Build neighborhood summaries from matches
   - `python build_neighborhood_from_matches.py`
4) Generate HTML maps
   - `python present_all_maps.py`
   - Open `outputs/interactive/index.html`
5) (Optional) Create house‑level master file enriched with tract race/income
   - `python master_file_creation.py`

## Notes
- All downloaders save three formats side‑by‑side in `data_raw/`: GeoJSON, CSV (with lon/lat or centroid), GeoParquet.
- Race tracts are filtered to Richmond by attribute (GEOID/NAME) and clipped to the neighborhood union before overlay to prevent county spillover.
- HTML maps display “No data” as grey; tooltips show name + value.

## Install
- pip: `pip install geopandas pandas requests pyarrow folium`
- conda: `conda install -c conda-forge geopandas pyarrow requests folium`

## Troubleshooting
- If tract maps render as points, ensure polygon tract layers exist in `data_raw/` (e.g., `RVA_Race_and_Ethnicity_2020.geoparquet`).
- If overlays drop features, scripts attempt geometry repair (`buffer(0)`) and use keep‑geom‑type overlays; check matches CSV + overlay HTML to diagnose.
