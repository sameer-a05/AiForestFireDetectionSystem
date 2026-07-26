import torch
import numpy as np
import rioxarray
from pathlib import Path
from torch.utils.data import Dataset

class WildfireDataSet(Dataset):
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        
        # 1. Grab the sorted lists from their specific subfolders
        stack_file = sorted(list((self.data_dir / "stacked").glob("*.tif")))
        mask_file = sorted(list((self.data_dir / "masks").glob("*.tif")))
        
        # 2. Zip them side-by-side into a list of tuples
        self.file_paths = list(zip(stack_file, mask_file))
        
    def __len__(self):
        # Tells PyTorch how many total items exist
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # 3. Unpack the exact stack and mask paths requested by the index
        stack_path, mask_path = self.file_paths[idx]
         
        # 4. Load using standard .values and immediately cast to float32 to avoid PyTorch uint16 errors
        stack_numpy = rioxarray.open_rasterio(stack_path).values.astype(np.float32)
        mask_numpy = rioxarray.open_rasterio(mask_path).values.astype(np.float32)
        
        # 5. Convert the raw numpy arrays into PyTorch Tensors
        image_tensor = torch.tensor(stack_numpy, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_numpy, dtype=torch.float32)
        
        # 6. Serve the meal!
        return (image_tensor, mask_tensor)

if __name__ == "__main__":
    # Robust pathing relative to the script's physical location
    script_dir = Path(__file__).parent
    processed_dir = script_dir.parent / "data" / "processed" / "dixie_2021"
    
    # Initialize your shiny new dataset
    dataset = WildfireDataSet(data_dir=processed_dir)
    
    # Call index 0 just like a standard list
    image, mask = dataset[0]
    
    print("--- Checkpoint Verification ---")
    print(f"Total pairs found: {len(dataset)}")
    print(f"Image tensor shape: {image.shape}")
    print(f"Mask tensor shape: {mask.shape}")
    
    # Check that the mask is strictly binary (only 0.0 and 1.0)
    unique_mask_values = torch.unique(mask)
    print(f"Unique mask values: {unique_mask_values.tolist()}")
    print("-------------------------------")