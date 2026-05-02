# %% Updated JSRT Dataset with Proper Preprocessing
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2

class JSRTDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform
        
        # Create binary labels: nodule (1) if state is malignant/benign, else non-nodule (0)
        self.df['label'] = self.df['state'].apply(
            lambda x: 1 if pd.notna(x) and x in ['malignant', 'benign'] else 0
        )
        
        print(f"JSRT Dataset loaded:")
        print(f"  Total samples: {len(self.df)}")
        print(f"  Nodules: {self.df['label'].sum()}")
        print(f"  Non-nodules: {(self.df['label'] == 0).sum()}")
        
    def __len__(self):
        return len(self.df)
    
    def preprocess_cxr(self, img_array):
        """
        Apply similar preprocessing as NODE21 standardization:
        1. Normalize intensities to [0, 255]
        2. Apply CLAHE for better contrast (simulates intensity standardization)
        3. Convert to 3-channel
        """
        # Ensure float32
        img_array = img_array.astype(np.float32)
        
        # Normalize to 0-255 range
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_array = np.zeros_like(img_array, dtype=np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This simulates the intensity standardization from opencxr
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_array = clahe.apply(img_array)
        
        return img_array
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['study_id'])
        
        # Load PNG image as grayscale
        image = Image.open(img_path).convert('L')  # Load as grayscale
        img_array = np.array(image)
        
        # Apply CXR-like preprocessing
        img_array = self.preprocess_cxr(img_array)
        
        # Convert to 3-channel (RGB) - IMPORTANT: Same as NODE21 processing
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
        
        # Convert back to PIL Image for transforms
        image = Image.fromarray(img_array)
        
        # Apply transforms (includes resize to 224x224)
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return image, label