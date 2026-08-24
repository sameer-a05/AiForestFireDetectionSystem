import numpy as np
import rioxarray
import random
from pathlib import Path

def slice_into_patches(stack_arr, mask_arr, tile_size=256):
    channels, max_h, max_w = stack_arr.shape
    stack_tiles = []
    mask_tiles = []
    
    # Outer loop (y-axis / height)
    for y in range(0, max_h, tile_size):
        # Inner loop (x-axis / width)
        for x in range(0, max_w, tile_size):
            stack_patch = stack_arr[:, y:y+tile_size, x:x+tile_size]
            mask_patch = mask_arr[:, y:y+tile_size, x:x+tile_size]

            # Use _ for unused variables
            _, patch_height, patch_width = stack_patch.shape
            
            # 1. Edge filter
            if patch_height != tile_size or patch_width != tile_size:
                continue
            
            # 2. Background filter
            if mask_patch.sum() == 0:
                roll = random.random()
                if roll > 0.30:
                    continue
            
            stack_tiles.append(stack_patch)
            mask_tiles.append(mask_patch)
            
    return stack_tiles, mask_tiles


if __name__ == "__main__":
    # Fix reproducibility 
    random.seed(42)

    # Set up paths relative to the script location using .parent
    script_dir = Path(__file__).parent
    stacked_dir = script_dir.parent / "data" / "processed" / "dixie_2021" / "stacked"
    masks_dir = script_dir.parent / "data" / "processed" / "dixie_2021" / "masks"
    
    output_stack_tile = script_dir.parent / "data" / "processed" / "dixie_2021" / "tiles" / "images"
    output_mask_tile = script_dir.parent / "data" / "processed" / "dixie_2021" / "tiles" / "masks"

    mask_tif = masks_dir / "mask_20210927.tif"
    stack_tif = stacked_dir / "stack_20210927.tif"

    # Load arrays at execution time, not import time
    print("Loading raster arrays...")
    stack_arr = rioxarray.open_rasterio(stack_tif).values
    mask_arr = rioxarray.open_rasterio(mask_tif).values

    print("Slicing image into patches...")
    final_stacks, final_masks = slice_into_patches(stack_arr, mask_arr)
    
    print(f"Total valid patches extracted: {len(final_stacks)}")
    
    # Physically create the output folders
    output_stack_tile.mkdir(parents=True, exist_ok=True)
    output_mask_tile.mkdir(parents=True, exist_ok=True)
    
    print("Saving patches to disk...")
    # Loop through the lists and save
    for idx in range(len(final_stacks)):
        stack_patch = final_stacks[idx]
        mask_patch = final_masks[idx]
        
        # Zero-pad the filenames to 4 digits (e.g., patch_0001.npy) to guarantee correct sorting
        stack_filename = output_stack_tile / f"patch_{idx:04d}.npy"
        mask_filename = output_mask_tile / f"patch_{idx:04d}.npy"
        
        np.save(stack_filename, stack_patch)
        np.save(mask_filename, mask_patch)
        
