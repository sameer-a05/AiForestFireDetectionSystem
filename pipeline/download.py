import planetary_computer
import pystac_client
import rioxarray
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

# 1. Define your region — Dixie Fire area, Northern California
bbox = [-121.5, 39.8, -120.5, 40.5]  # [min_lon, min_lat, max_lon, max_lat]

# 2. Define your time window
time_window = "2021-08-01/2021-09-30"

# 3. Search the Sentinel-2 catalog
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=time_window,
    query={"eo:cloud_cover": {"lt": 20}}  # less than 20% cloud cover
)

items = list(search.items())
print(f"Found {len(items)} scenes")

# 4. Download and clip one scene to verify
item = items[0]
print(f"Scene date: {item.datetime.date()}")

# Load the Red band (B04) — fire shows up well in Red + NIR combination
ds = rioxarray.open_rasterio(
    planetary_computer.sign(item.assets["B04"].href),
    masked=True
).squeeze()

# 5. Clip to bounding box
ds_clipped = ds.rio.clip_box(*bbox)

# 6. Save as GeoTIFF
ds_clipped.rio.to_raster("scene_20210815_B04.tif")
print("Saved GeoTIFF")

# 7. Visualize to confirm
plt.figure(figsize=(8, 6))
plt.imshow(ds_clipped.values, cmap="gray")
plt.title(f"Sentinel-2 Band 4 — {item.datetime.date()}")
plt.colorbar()
plt.savefig("preview.png")
plt.show()
