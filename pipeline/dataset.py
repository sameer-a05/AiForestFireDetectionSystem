import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class WildfireDataSet(Dataset):
    def __init__(self, data_dir: str | Path, indices=None):
        self.data_dir = Path(data_dir)
        
        # 1. Grab the sorted lists from their specific subfolders (looking for .npy now!)
        stack_file = sorted(list((self.data_dir / "images").glob("*.npy")))
        mask_file = sorted(list((self.data_dir / "masks").glob("*.npy")))
        
        # 2. Zip them side-by-side into a list of tuples
        self.file_paths = list(zip(stack_file, mask_file))
        
        # 3. Apply the train/val split filter if indices were provided
        if indices is not None:
            self.file_paths = [self.file_paths[i] for i in indices]
        
    def __len__(self):
        # Tells PyTorch how many items exist in this specific split
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # 4. Unpack the exact stack and mask paths requested by the index
        stack_path, mask_path = self.file_paths[idx]
         
        # 5. Load using np.load() instead of rioxarray, cast to float32
        stack_numpy = np.load(stack_path)
        mask_numpy = np.load(mask_path)
        
        # 6. Convert the raw numpy arrays into PyTorch Tensors
        image_tensor = torch.tensor(stack_numpy, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_numpy, dtype=torch.float32)
        
        return (image_tensor, mask_tensor)

if __name__ == "__main__":
    # Robust pathing relative to the script's physical location
    script_dir = Path(__file__).parent
    processed_dir = script_dir.parent / "data" / "processed" / "dixie_2021" / "tiles"
    
    # --- The Split Logic ---
    # Find out how many total tile pairs we have
    image_files = list((processed_dir / "images").glob("*.npy"))
    mask_files = list((processed_dir / "masks").glob("*.npy"))
    
    assert len(image_files) == len(mask_files), f"Mismatch: {len(image_files)} images vs {len(mask_files)} masks"
    
    total_files = len(image_files)
    
    # Create a list of all index numbers [0, 1, 2, ..., total_files - 1]
    all_indices = list(range(total_files))
    
    # Shuffle the indices (with a seed so your split is identical every time you run it)
    np.random.seed(42)
    np.random.shuffle(all_indices)
    
    # Calculate the 80% cutoff point
    split_point = int(total_files * 0.8)
    
    # Slice the shuffled list into Train (80%) and Val (20%)
    train_indices = all_indices[:split_point]
    val_indices = all_indices[split_point:]
    
    # Initialize the two datasets
    train_dataset = WildfireDataSet(data_dir=processed_dir, indices=train_indices)
    val_dataset = WildfireDataSet(data_dir=processed_dir, indices=val_indices)
    
    # Call index 0 on the training set to verify shapes
    image, mask = train_dataset[0]
    
    print("--- Checkpoint Verification ---")
    print(f"Total tiles on disk: {total_files}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Image tensor shape: {image.shape}")
    print(f"Mask tensor shape: {mask.shape}")
    
    unique_mask_values = torch.unique(mask)
    print(f"Unique mask values: {unique_mask_values.tolist()}")
    print("-------------------------------")