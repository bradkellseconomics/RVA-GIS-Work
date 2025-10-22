import os
import sys
from typing import Optional, List

import geopandas as gpd
import pandas as pd


DERIVED_DIR = "data_derived"
OUT_DIR = os.path.join("outputs", "interactive")


def log(msg: str):
    print(msg, flush=True)


def load_enriched() -> Optional[gpd.GeoDataFrame]:
    base = os.path.join(DERIVED_DIR, "neighborhoods_enriched")
    for ext in (".geoparquet", ".geojson", ".parquet"):
        path = base + ext
        if os.path.exists(path):
            try:
                if ext in (".geoparquet", ".parquet"):
                    return gpd.read_parquet(path)
                return gpd.read_file(path)
            except Exception as e:
                log(f"[WARN] Failed to read {path}: {e}")
    return None


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        if gdf.crs is None:
            return gdf.set_crs(epsg=4326)
        return gdf.to_crs(epsg=4326)
    except Exception:
        return gdf


def make_colormap(series: pd.Series):
    from branca.colormap import LinearColormap
    s = pd.to_numeric(series, errors="coerce")
    # If values look like proportions, scale to percentage for legend clarity
    vmax_q = s.quantile(0.95)
    scaled = False
    if pd.notna(vmax_q) and vmax_q <= 1.0:
        s = s * 100.0
        scaled = True
    vmin = float(s.quantile(0.05)) if s.notna().any() else 0.0
    vmax = float(s.quantile(0.95)) if s.notna().any() else 1.0
    if vmin == vmax:
        vmax = vmin + (1.0 if vmin == 0 else abs(vmin) * 0.1)
    cmap = LinearColormap(["#fee8c8", "#fdbb84", "#e34a33"], vmin=vmin, vmax=vmax)
    return cmap, scaled


def choropleth(gdf: gpd.GeoDataFrame, column: str, title: Optional[str], out_html: str, name_field: Optional[str]):
    try:
        import folium
    except ImportError:
        log("[ERROR] folium is required. Install with: pip install folium")
        return

    g = ensure_wgs84(gdf).dropna(subset=["geometry"])  # type: ignore[arg-type]
    if g.empty:
        log(f"[WARN] No geometry to plot for {column}")
        return

    # Center
    try:
        minx, miny, maxx, maxy = g.total_bounds
        lat = (miny + maxy) / 2
        lon = (minx + maxx) / 2
    except Exception:
        lat, lon = 37.5407, -77.4360

    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")
    cmap, scaled = make_colormap(g[column])

    # Tooltip fields
    fields: List[str] = []
    aliases: List[str] = []
    if name_field and name_field in g.columns:
        fields.append(name_field)
        aliases.append("Neighborhood")
    fields.append(column)
    aliases.append((title or column) + (" (%)" if scaled else ""))

    def style_fn(feat):
        val = feat["properties"].get(column)
        try:
            vnum = float(val) * (100.0 if scaled else 1.0) if val is not None else None
        except Exception:
            vnum = None
        col = cmap(vnum) if vnum is not None else "#cccccc"
        return {"fillColor": col, "color": "#666", "weight": 0.8, "fillOpacity": 0.7}

    folium.GeoJson(
        g.to_json(default=str),
        name=title or column,
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True),
    ).add_to(m)

    cmap.caption = title or column
    m.add_child(cmap)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    log(f"[OK] Wrote {out_html}")


def main():
    gdf = load_enriched()
    if gdf is None or gdf.empty:
        log(f"[ERROR] Could not load neighborhoods_enriched from {DERIVED_DIR}")
        sys.exit(1)

    # Pick name column
    name_col = None
    for c in ["Name", "NAME", "neighborhood", "Neighborhood", "neighborhood_name"]:
        if c in gdf.columns:
            name_col = c
            break

    # Lead metrics
    lead_cols = [c for c in ["pct_known_lead", "pct_unknown", "pct_known_nonlead", "pct_known_lead_total", "pct_known_nonlead_total"] if c in gdf.columns]
    for c in lead_cols:
        choropleth(gdf, c, title=c.replace("_", " ").title(), out_html=os.path.join(OUT_DIR, f"neighborhoods_enriched_{c}.html"), name_field=name_col)

    # Race metrics (only map those present)
    race_cols = [c for c in ["pct_black", "pct_white", "pct_hispanic", "pct_asian", "pct_native", "pct_hipi"] if c in gdf.columns]
    for c in race_cols:
        choropleth(gdf, c, title=c.replace("pct_", "Pct ").replace("_", " ").title(), out_html=os.path.join(OUT_DIR, f"neighborhoods_enriched_{c}.html"), name_field=name_col)

    # Simple index page linking outputs
    items = []
    for fn in sorted(os.listdir(OUT_DIR)):
        if fn.startswith("neighborhoods_enriched_") and fn.endswith(".html"):
            items.append(fn)
    if items:
        index_path = os.path.join(OUT_DIR, "neighborhoods_enriched_index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("<h2>Neighborhoods Enriched Maps</h2>\n<ul>\n")
            for fn in items:
                f.write(f"  <li><a href='{fn}' target='_blank'>{fn}</a></li>\n")
            f.write("</ul>\n")
        log(f"[OK] Wrote index -> {index_path}")


if __name__ == "__main__":
    main()
