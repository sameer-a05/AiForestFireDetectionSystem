import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray
import rasterio.features
import numpy as np
from pathlib import Path
import xarray as xr

# 1. setup paths and dirs
script_dir = Path(__file__).parent
dixie_output_dir = script_dir.parent / "data" / "processed" / "dixie_2021"
geoTIFFs_dir = script_dir.parent / "data" / "raw" / "sentinel2"

#  Ensure the output masks directory exists
masks_dir = dixie_output_dir / "masks"
masks_dir.mkdir(parents=True, exist_ok=True)
stacked_dir = dixie_output_dir / "stacked"
stacked_dir.mkdir(parents=True,exist_ok=True)

# file paths
B04geoTIFFs_file_path = geoTIFFs_dir / "scene_20210926_B04_merged.tif"
B08geoTIFFs_file_path = geoTIFFs_dir / "scene_20210926_B08_merged.tif"
B12geoTIFFs_file_path = geoTIFFs_dir / "scene_20210926_B12_merged.tif"
dixie_file_path = dixie_output_dir / "20210927_Dixie_IR Dixie.shp"

output_mask_path = masks_dir / "mask_20210927.tif"
output_stacked_path = stacked_dir / "stack_20210927.tif"


# stacking the satellite bands
print("Processing Satellite")
geoTIFFs_paths = [B04geoTIFFs_file_path,B08geoTIFFs_file_path,B12geoTIFFs_file_path]

#load template to force matching grids
template_band = rioxarray.open_rasterio(geoTIFFs_paths[0])
processed_bands = []

#load, force it to reproject b04 grid, normalize, squeeze
for path in geoTIFFs_paths:
    band = rioxarray.open_rasterio(path)
    band = band.rio.reproject_match(template_band)
    band_norm = (band - band.min()) / (band.max() - band.min())
    processed_bands.append(band_norm.squeeze())
    
stacked_array = np.stack(processed_bands)
    
print(f"Stacked Array Shape: {stacked_array.shape}")
print(f"Min value: {stacked_array.min()}")
print(f"Max value: {stacked_array.max()}")

#saving stacked array within stacked folder
stacked_da = xr.DataArray(
    stacked_array,
    coords={"band": [1,2,3], "y": template_band.y, "x":template_band.x},
    dims=("band", "y", "x")
)

stacked_da.rio.write_crs(template_band.rio.crs, inplace =True)
stacked_da.rio.to_raster(output_stacked_path)
print(f"Successfully saved stacked image to: {output_stacked_path}\n")


#burning the mask
gdf = gpd.read_file(dixie_file_path)

raster_data = rioxarray.open_rasterio(B04geoTIFFs_file_path)

projected_gdf = gdf.to_crs(raster_data.rio.crs)

# Get the transform (spatial coordinates) and shape (pixel dimensions) of the GeoTIFF
transform = raster_data.rio.transform()
out_shape = (raster_data.rio.height, raster_data.rio.width)

#  Extract the geometry column as a list of shapes
fire_shapes = projected_gdf.geometry.tolist()

# Burn the perimeter polygons onto the imagery grid
# print("Burning mask onto grid...")
mask_array = rasterio.features.rasterize(
    shapes=fire_shapes,
    out_shape=out_shape,
    transform=transform,
    fill=0,           # Pixels outside the perimeter = 0
    default_value=1,  # Pixels inside the perimeter = 1
    dtype=np.uint8
)

# Print the unique values to verify
unique_values = np.unique(mask_array)
print(f"Unique values in output array: {unique_values}")

# Save the mask as a GeoTIFF
# Copy the original xarray object to keep all the spatial metadata perfectly intact,
# then replace its pixel values with our new 0s and 1s mask.
mask_da = raster_data.copy()
mask_da.values[0] = mask_array 

mask_da.rio.to_raster(output_mask_path)
print(f"Successfully saved mask to: {output_mask_path}")


