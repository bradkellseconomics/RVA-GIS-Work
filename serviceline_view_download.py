import math, time, os, requests, geopandas as gpd
from urllib.parse import urlencode

BASE = ("https://services1.arcgis.com/k3vhq11XkBNeeOfM/arcgis/rest/services/"
        "ServiceLine_viewing_3caed2823fff46b8b08d808720794506/FeatureServer/0")

# 1) service limits + total rows
meta = requests.get(f"{BASE}?f=pjson", timeout=30).json()
mrc = meta.get("maxRecordCount", 2000)

total = requests.get(
    f"{BASE}/query",
    params={"where":"1=1","returnCountOnly":"true","f":"pjson"},
    timeout=60
).json()["count"]

pages = math.ceil(total / mrc)
print(f"maxRecordCount={mrc}, total={total}, pages={pages}")

# 2) page through results
frames = []
for i in range(pages):
    offset = i * mrc
    params = {
        "where": "1=1",
        "outFields": "*",
        "outSR": 4326,            # lon/lat
        "f": "geojson",
        "resultOffset": offset,
        "resultRecordCount": mrc,
        "returnGeometry": "true",
        "resultType": "standard"  # avoid tile/quantized responses
    }
    url = f"{BASE}/query?{urlencode(params)}"
    print(f"Page {i+1}/{pages}")
    frames.append(gpd.read_file(url))
    time.sleep(0.25)  # be polite

gdf = gpd.pd.concat(frames, ignore_index=True)
print("Downloaded:", gdf.shape)

# 3) convenience columns + saves
gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y
out_dir = os.path.join("data_raw")
os.makedirs(out_dir, exist_ok=True)
geojson_path = os.path.join(out_dir, "richmond_ServiceLine_view.geojson")
csv_path = os.path.join(out_dir, "richmond_ServiceLine_view.csv")
geopq_path = os.path.join(out_dir, "richmond_ServiceLine_view.geoparquet")
gdf.to_file(geojson_path, driver="GeoJSON")
gdf.drop(columns="geometry").to_csv(csv_path, index=False)
try:
    gdf.to_parquet(geopq_path, index=False)
except Exception as e:
    print(f"[WARN] Failed to write GeoParquet: {e}")
print(f"Saved GeoJSON to: {geojson_path}")
print(f"Saved CSV to: {csv_path}")
print(f"Saved GeoParquet to: {geopq_path}")
