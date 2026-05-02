"""
Nodule21 Dataset for Nodule Binary Classification
Handles chest X-ray images with nodule presence detection
"""

import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

class NoduleDataset(Dataset):
    def __init__(self, dataframe, images_dir, transform=None, return_bbox=False, target_size=224):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.return_bbox = return_bbox
        self.target_size = target_size  # Add this!
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['img_name'])
        
        # Load .mha image
        image = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(image)
        
        # Get original image size
        if len(img_array.shape) == 3:
            img_array = img_array[0]
        original_height, original_width = img_array.shape  # e.g., 1024×1024
        
        # Normalize to 0-255 range
        img_array = img_array.astype(np.float32)
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_array = np.zeros_like(img_array, dtype=np.uint8)
        
        # Convert to 3-channel (RGB)
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
        
        # Convert to PIL Image for transforms
        from PIL import Image
        image = Image.fromarray(img_array)
        
        if self.transform:
            image = self.transform(image)  # This resizes to 224×224
        
        label = torch.tensor(row['label'], dtype=torch.long)
        
        if self.return_bbox:
            # Scale bbox coordinates from original size to target size
            scale_x = self.target_size / original_width
            scale_y = self.target_size / original_height
            
            x_scaled = row['x'] * scale_x
            y_scaled = row['y'] * scale_y
            w_scaled = row['width'] * scale_x
            h_scaled = row['height'] * scale_y
            
            # NOW normalize to [0, 1] based on TARGET size (224)
            x_norm = x_scaled / self.target_size
            y_norm = y_scaled / self.target_size
            w_norm = w_scaled / self.target_size
            h_norm = h_scaled / self.target_size
            
            bbox = torch.tensor([x_norm, y_norm, w_norm, h_norm], dtype=torch.float32)
            return image, label, bbox
        
        return image, label