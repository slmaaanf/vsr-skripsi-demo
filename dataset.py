import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class VSRDataset(Dataset):
    def __init__(self, root_dir, csv_file, mode, class_to_idx=None):
        """
        Args:
            root_dir (string): Directory with all the .npy files.
            csv_file (string): Path to the csv file with annotations.
            mode (string): 'public' or 'private' (not used strictly here, but for compatibility).
            class_to_idx (dict): Dictionary mapping text label to integer index.
        """
        self.root_dir = root_dir
        # Handle delimiter automatically
        try:
            self.data = pd.read_csv(csv_file, sep='|', header=None, names=['fn', 'text'])
        except:
            self.data = pd.read_csv(csv_file)
            
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Ambil Nama File & Label
        filename = str(self.data.iloc[idx, 0])
        text_label = str(self.data.iloc[idx, 1]).strip()
        
        # 2. Load NPY
        file_path = os.path.join(self.root_dir, filename)
        
        try:
            # Data di NPY sudah (Time, Height, Width) dan Normalized 0-1 (Float32)
            video_array = np.load(file_path)
        except Exception as e:
            # Fallback jika file rusak (return tensor kosong biar gak crash total)
            print(f"Error loading {file_path}: {e}")
            return torch.zeros((1, 1, 75, 50, 100)), 0

        # 3. Convert to Tensor
        # Input Model butuh: (Channel, Time, Height, Width)
        # Data NPY saat ini: (Time, Height, Width)
        tensor = torch.FloatTensor(video_array)
        tensor = tensor.unsqueeze(0) # Tambah Channel dim -> (1, T, H, W)
        
        # 4. Convert Label to ID
        if self.class_to_idx:
            label_id = self.class_to_idx.get(text_label, 0) # Default 0 if not found
        else:
            label_id = 0
            
        return tensor, torch.tensor(label_id, dtype=torch.long)