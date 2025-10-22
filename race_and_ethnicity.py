# rva_download_all_layers.py
import requests, json, geopandas as gpd
from urllib.parse import urlencode
from pathlib import Path

ITEM_ID = "092465126da1446b9722c10357b83cd5"  # RVA Race & Ethnicity (Web Map)
OUTDIR = Path("rva_layers")
OUTDIR.mkdir(exist_ok=True)
RAW_OUTDIR = Path("data_raw")
RAW_OUTDIR.mkdir(exist_ok=True)

def get_item_info(item_id):
    r = requests.get(f"https://www.arcgis.com/sharing/rest/content/items/{item_id}",
                     params={"f":"pjson"}, timeout=30)
    r.raise_for_status()
    return r.json()

def get_webmap_layers(item_id):
    """Return list of dicts with name + url for each operational layer in a Web Map."""
    r = requests.get(f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/data",
                     params={"f":"json"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    layers = []
    for ol in data.get("operationalLayers", []):
        url = ol.get("url")
        name = ol.get("title") or ol.get("layerType") or "layer"
        if url:
            layers.append({"name": name, "url": url})
        # Some web maps nest sublayers under "layers"
        for sub in ol.get("layers", []) or []:
            if sub.get("url"):
                layers.append({"name": sub.get("title") or name, "url": sub["url"]})
    return layers

def get_service_layers(service_url):
    """Return list of dicts with name + url for each layer index in a Feature/Map Service."""
    meta = requests.get(service_url, params={"f":"pjson"}, timeout=30).json()
    layers = []
    for lyr in meta.get("layers", []):
        idx = lyr["id"]
        name = lyr.get("name", f"layer_{idx}")
        layers.append({"name": name, "url": f"{service_url}/{idx}"})
    return layers

def _sanitize(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_")


def download_layer_geojson_to_parquet(layer_url, name):
    params = {
        "where": "1=1",
        "outFields": "*",
        "outSR": 4326,
        "returnGeometry": "true",
        "resultType": "standard",
        "f": "geojson",
    }
    # First get count + maxRecordCount for paging
    base = layer_url.rstrip("/")
    meta = requests.get(base.split("/FeatureServer/")[0] + "/FeatureServer",
                        params={"f": "pjson"}, timeout=30).json()
    # If above split fails (MapServer), just fetch layer's own meta:
    lyr_meta = requests.get(base, params={"f": "pjson"}, timeout=30).json()
    mrc = lyr_meta.get("maxRecordCount", meta.get("maxRecordCount", 2000))

    cnt = requests.get(base + "/query",
                       params={"where":"1=1","returnCountOnly":"true","f":"pjson"},
                       timeout=30).json().get("count", 0)

    frames = []
    offset = 0
    while offset < cnt:
        page = params | {"resultOffset": offset, "resultRecordCount": mrc}
        url = f"{base}/query?{urlencode(page)}"
        gdf = gpd.read_file(url)
        if len(gdf) == 0:
            break
        frames.append(gdf)
        offset += mrc
    if not frames:
        print(f"[WARN] No features in {name}")
        return
    import pandas as pd
    gdf_all = pd.concat(frames, ignore_index=True)
    # Ensure geometry present if any
    if "geometry" in getattr(gdf_all, "columns", []):
        gdf_all = gpd.GeoDataFrame(gdf_all, geometry="geometry", crs="EPSG:4326")
    # Write unified raw GeoJSON, CSV mirrors, and GeoParquet in data_raw
    base = _sanitize(name)
    geojson_path = RAW_OUTDIR / f"{base}.geojson"
    csv_path = RAW_OUTDIR / f"{base}.csv"
    geopq_path = RAW_OUTDIR / f"{base}.geoparquet"
    try:
        gdf_all.to_file(geojson_path, driver="GeoJSON")
    except Exception as e:
        print(f"[WARN] Failed to write GeoJSON for {name}: {e}")
    try:
        gdf_all.to_parquet(geopq_path, index=False)
    except Exception as e:
        print(f"[WARN] Failed to write GeoParquet for {name}: {e}")
    try:
        df_csv = gdf_all.copy()
        if df_csv.crs is None:
            df_csv = df_csv.set_crs(epsg=4326)
        # Add lon/lat or centroid coords
        try:
            geom_types = set(df_csv.geometry.geom_type.dropna().unique().tolist()) if hasattr(df_csv, "geometry") else set()
            if geom_types.issubset({"Point", "MultiPoint"}):
                pts = df_csv.to_crs(epsg=4326)
                df_csv["lon"] = pts.geometry.x
                df_csv["lat"] = pts.geometry.y
            elif hasattr(df_csv, "geometry"):
                proj = df_csv.to_crs(epsg=3857)
                cent = proj.geometry.centroid.to_crs(epsg=4326)
                df_csv["centroid_lon"] = cent.x
                df_csv["centroid_lat"] = cent.y
        except Exception:
            pass
        if "geometry" in getattr(df_csv, "columns", []):
            df_csv = df_csv.drop(columns=["geometry"])  # type: ignore[arg-type]
        df_csv.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"[WARN] Failed to write CSV for {name}: {e}")

    print(f"[OK] Saved raw -> {geojson_path}, {csv_path}, {geopq_path} ({len(gdf_all)} features)")

info = get_item_info(ITEM_ID)
typ = info.get("type", "")
print("Item type:", typ)

layers_to_fetch = []

if "Web Map" in typ:
    # Discover operational layers in the web map
    wm_layers = get_webmap_layers(ITEM_ID)
    if not wm_layers:
        raise SystemExit("No operational layers found in the Web Map.")
    # Each url here is usually a FeatureServer layer root or a service root; normalize to layer URLs
    for L in wm_layers:
        url = L["url"]
        if "/FeatureServer/" in url or "/MapServer/" in url:
            # If it's a service root (endswith /FeatureServer or /MapServer), enumerate sublayers:
            if url.rstrip("/").endswith(("FeatureServer", "MapServer")):
                sublayers = get_service_layers(url)
                layers_to_fetch.extend(sublayers)
            else:
                layers_to_fetch.append({"name": L["name"], "url": url})
elif "Feature Service" in typ or "Feature Layer" in typ or "Map Service" in typ:
    service_url = info.get("url")
    if not service_url:
        raise SystemExit("Service item missing URL.")
    layers_to_fetch = get_service_layers(service_url)
else:
    raise SystemExit(f"Unsupported item type: {typ}")

# Deduplicate by URL
seen = set()
unique_layers = []
for L in layers_to_fetch:
    if L["url"] not in seen:
        unique_layers.append(L)
        seen.add(L["url"])

print(f"Discovered {len(unique_layers)} layer(s). Downloading…")
for L in unique_layers:
    download_layer_geojson_to_parquet(L["url"], L["name"])
