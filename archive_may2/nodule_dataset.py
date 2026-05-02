import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path


class NoduleDataset(Dataset):
    """Dataset for lung nodule classification"""
    
    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file with annotations
            image_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Create label mapping
        self.label_map = {
            'malignant': 0,
            'benign': 1,
            'non-nodule': 2
        }
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image path and label
        img_name = self.df.iloc[idx]['study_id']
        img_path = self.image_dir / img_name
        state = self.df.iloc[idx]['state']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.label_map[state]
        
        return image, label
    
    def get_class_counts(self):
        """Returns counts for each class"""
        state_counts = self.df['state'].value_counts()
        return {
            'malignant': state_counts.get('malignant', 0),
            'benign': state_counts.get('benign', 0),
            'non-nodule': state_counts.get('non-nodule', 0)
        }