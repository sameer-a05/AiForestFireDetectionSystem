import planetary_computer
import pystac_client
import rioxarray
import matplotlib.pyplot as plt
from rioxarray.merge import merge_arrays
from pathlib import Path

def data_pipeline():
    # 1. Define your region — Dixie Fire area, Northern California
    bbox = [-121.5, 39.8, -120.5, 40.5]  # [min_lon, min_lat, max_lon, max_lat]
    
    # 2. Define your time window
    time_window = "2021-08-01/2021-09-30"
    
    # empty dict for hash map
    daily_items = {}
    
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
    
    for item in search.items():
        # 1. Extract the YYYY-MM-DD string from the item's built-in timestamp
        date_str = item.datetime.strftime("%Y-%m-%d")
        
        # 2. If this date isn't in our dictionary yet, create a new list for it
        if date_str not in daily_items:
            daily_items[date_str] = []
            
        # 3. Append the full item object to that date's list
        daily_items[date_str].append(item)

    # target certain day for mosaicking (Dynamic Date Selection)
    target_date = max(daily_items, key=lambda k: len(daily_items[k]))
    target_items = daily_items[target_date]
    print(f"Selected optimal date: {target_date} with {len(target_items)} tiles.")

    rasters_to_merge = []
    
    # loop open, prep, and store each tile
    for item in target_items:
        signed_href = planetary_computer.sign(item.assets["B04"].href)
        ds = rioxarray.open_rasterio(
            signed_href, 
            masked=True
        ).squeeze()
        
        rasters_to_merge.append(ds)

    # stitch the 3 arrays together using their internal coordinates
    merged_ds = merge_arrays(rasters_to_merge)

    # clip the newly stitched, massive dataset to your exact bounding box
    final_clipped = merged_ds.rio.clip_box(*bbox, crs="EPSG:4326")

    # Dynamic File Pathing setup
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "raw" / "sentinel2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"scene_{target_date.replace('-', '')}_B04_merged.tif"
    output_path = output_dir / filename
    
    # Export to GeoTIFF
    final_clipped.rio.to_raster(output_path)
    print(f"Saved GeoTIFF to: {output_path}")

    return final_clipped, target_date


def visualize(clip, date_string):
    plt.figure(figsize=(8, 6))
    plt.imshow(clip.values, cmap="gray")
    plt.title(f"Sentinel-2 Band 4 — {date_string}")
    plt.colorbar()
    plt.show()


clip, optimal_date = data_pipeline()
visualize(clip, optimal_date)