import matplotlib.pyplot as plt
import rioxarray
from pathlib import Path

# 1. Define your paths (pointing to the files you created in Tasks 4 & 5)
script_dir = Path(__file__).parent
processed_dir = script_dir.parent / "data" / "processed" / "dixie_2021"

stack_path = processed_dir / "stacked" / "stack_20210927.tif"
mask_path = processed_dir / "masks" / "mask_20210927.tif"

# 2. Load the data using rioxarray
stack_data = rioxarray.open_rasterio(stack_path)
mask_data = rioxarray.open_rasterio(mask_path)

# 3. Create the side-by-side canvas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# 4. Plot the B04 Band (Layer 0) on the left screen (ax1) in grayscale
stack_data[0].plot.imshow(ax=ax1, cmap='gray')

# 5. Plot the Mask on the right screen (ax2) in red
mask_data[0].plot.imshow(ax=ax2, cmap='Reds')

# Add titles and show the result 
ax1.set_title("Sentinel-2: B04 (Red Band)")
ax2.set_title("Ground Truth: Fire Mask")
plt.tight_layout()
plt.show()