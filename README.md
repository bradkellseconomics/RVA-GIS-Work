# Richmond GIS: Downloads, Aggregation, Lead, and Maps

Lightweight scripts to download Richmond GIS datasets, aggregate tract data to neighborhoods, build a house-level master file, run lead imputation/analysis, and produce interactive and printable maps. The workflow is modular and reproducible.

## Data Layout
- `data_raw/` - raw inputs (GeoJSON/CSV/GeoParquet) written by download scripts
- `data_derived/` - derived outputs (matches, neighborhood summaries)
- `data/analysis/` - analysis-ready CSV/Parquet (master file, EDA, imputation)
- `data/regression/` - regression datasets and model outputs
- `outputs/interactive/` - HTML maps
- `outputs/presenting/` - PDF maps

## Scripts
- Downloads
  - `download_all.py` - orchestrator (GeoHub layers, service lines, race/ethnicity, income)
  - `get_geohub_files.py` - selected ArcGIS Hub datasets (e.g., neighborhoods, HS zones)
  - `serviceline_view_download.py` - Richmond ServiceLine view (houses)
  - `race_and_ethnicity.py` - race/ethnicity layers (tracts)
  - `download_income.py` - income layers (tracts)
- Tract-to-Neighborhood aggregation (recommended path)
  - `merge_tracts_to_neighborhoods.py` - builds tract–neighborhood matches via overlay; writes:
    - `data_derived/nbh_tract_matches_race.csv`
    - `data_derived/nbh_tract_matches_income.csv`
    - QA maps: `outputs/interactive/nbh_tract_overlay_{race|income}.html`
  - `build_neighborhood_from_matches.py` - computes neighborhood race (% shares) and income (weighted averages) using match fractions; writes:
    - `data_derived/neighborhoods_from_matches_race.{csv,geoparquet}`
    - `data_derived/neighborhoods_from_matches_income.{csv,geoparquet}`
- House-level master
  - `master_file_creation.py` - joins service lines to neighborhoods, and directly enriches houses with tract-level race and income; writes:
    - `data/analysis/servicelines_house_with_attributes.{csv,geoparquet}`
- Lead imputation + analysis
  - `imputing_lead.py` - cleans material fields, imputes unknowns from model classifications, and summarizes by neighborhood; writes:
    - `data/analysis/neighborhoods_lead_imputed_summary.csv`
    - `data/analysis/servicelines_with_imputed_materials.csv`
  - `map_imputed_lead_by_neighborhood.py` - maps imputed lead % by neighborhood; writes:
    - `outputs/interactive/neighborhoods_imputed_lead.html`
  - `analyze_servicelines_eda.py` - minimal EDA dataset and summaries; writes:
    - `data/analysis/servicelines_eda_minimal.csv`
    - `data/analysis/servicelines_eda_summary_{numeric,categorical}.csv`
  - `analyze_servicelines_logit.py` - logistic regression (age, post-1988, optional MHI, etc.); writes to `data/regression/`:
    - `servicelines_logit_age_dataset.csv`
    - `servicelines_logit_age_{terms,model}_statsmodels.csv` (if statsmodels available)
    - `servicelines_logit_age_{coefs,metrics,confusion}_sklearn.csv`
- Maps
  - `present_all_maps.py` - generates interactive maps (tract race/income, neighborhoods from matches, core layers) and index page
  - `presenting.py` - printable PDFs for core layers

## Workflow
1) Download raw data
   - `python download_all.py`
2) Build tract–neighborhood matches + QA
   - `python merge_tracts_to_neighborhoods.py`
   - Inspect: `data_derived/nbh_tract_matches_{race|income}.csv` and overlay HTML in `outputs/interactive/`
3) Build neighborhood summaries from matches
   - `python build_neighborhood_from_matches.py`
4) Generate HTML maps (core layers)
   - `python present_all_maps.py`
   - Open `outputs/interactive/index.html`
5) (Optional) Create house-level master file enriched with tract race/income
   - `python master_file_creation.py`
6) (Optional) Lead imputation, mapping, and regression
   - `python analyze_servicelines_eda.py`
   - `python imputing_lead.py`
   - `python map_imputed_lead_by_neighborhood.py` (writes `outputs/interactive/neighborhoods_imputed_lead.html`)
   - `python analyze_servicelines_logit.py`

## Notes
- All downloaders save three formats side-by-side in `data_raw/`: GeoJSON, CSV (with lon/lat or centroid), GeoParquet.
- Race tracts are filtered to Richmond by attribute (GEOID/NAME) and clipped to the neighborhood union before overlay to prevent county spillover.
- HTML maps display "No data" as grey; tooltips show name + value.

## Install
- pip: `pip install geopandas pandas requests pyarrow folium branca`
- conda: `conda install -c conda-forge geopandas pyarrow requests folium branca`
- Optional for regression: `pip install scikit-learn statsmodels`

## Troubleshooting
- If tract maps render as points, ensure polygon tract layers exist in `data_raw/` (e.g., `RVA_Race_and_Ethnicity_2020.geoparquet`).
- If overlays drop features, scripts attempt geometry repair (`buffer(0)`) and use keep-geom-type overlays; check matches CSV + overlay HTML to diagnose.
- If `map_imputed_lead_by_neighborhood.py` errors on imports, install `folium` and `branca`.
- If regression scripts warn about missing packages, install `statsmodels` and `scikit-learn`.

