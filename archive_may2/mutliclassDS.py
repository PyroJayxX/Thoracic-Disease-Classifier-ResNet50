#import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
import cv2
import torch

class NoduleDetectionDataset(Dataset):
    """Dataset for nodule detection + classification"""
    
    def __init__(self, csv_path, image_dir, transform=None, img_size=512):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Label mapping
        self.label_map = {
            'malignant': 0,
            'benign': 1,
            'non-nodule': 2
        }
        
        # Pre-compute paths and labels
        self.image_paths = [self.image_dir / row['study_id'] for _, row in self.df.iterrows()]
        self.labels = [self.label_map[row['state']] for _, row in self.df.iterrows()]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = image.shape
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image)
        
        # Get bounding box (normalized to 0-1)
        if pd.notna(row['x']) and pd.notna(row['y']):
            # Has nodule
            x = row['x'] / orig_w
            y = row['y'] / orig_h
            w = row['size'] / orig_w if pd.notna(row['size']) else 0.05  # Default size if missing
            h = row['size'] / orig_h if pd.notna(row['size']) else 0.05
            bbox = torch.tensor([x, y, w, h], dtype=torch.float32)
            has_nodule = 1.0
        else:
            # No nodule
            bbox = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            has_nodule = 0.0
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, bbox, label, has_nodule