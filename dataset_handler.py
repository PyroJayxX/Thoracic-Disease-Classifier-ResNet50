import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# ============================================================================
# CUSTOM DATASET CLASS (Modified for TARGET labels only)
# ============================================================================

class NIHChestXrayDataset(Dataset):
    """
    Custom Dataset for NIH Chest X-ray images
    Returns only the TARGET labels for prediction
    """
    
    def __init__(self, dataframe, image_dir, target_labels, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.target_labels = target_labels  # Only the 5 classes we predict
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.image_dir, self.dataframe.iloc[idx]['Path'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get labels - ONLY for target classes (5 classes)
        labels = torch.tensor(
            self.dataframe.iloc[idx][self.target_labels].values.astype(np.float32)
        )
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, labels