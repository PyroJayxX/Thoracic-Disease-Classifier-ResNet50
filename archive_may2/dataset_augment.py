import pandas as pd
import numpy as np
import os
from pathlib import Path
import shutil
from PIL import Image
from scipy.ndimage import rotate, shift
import SimpleITK as sitk

# Paths
base_path = "./dataset_nodule21/cxr_images/proccessed_data"
images_path = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "metadata.csv")
output_images_path = os.path.join(base_path, "new_images")
output_csv_path = os.path.join(base_path, "metadata_upsampled.csv")

# Create output directory
os.makedirs(output_images_path, exist_ok=True)

# Read metadata
df = pd.read_csv(csv_path)

# Separate positive and negative samples
positive_df = df[df['label'] == 1].copy()
negative_df = df[df['label'] == 0].copy() if 0 in df['label'].values else pd.DataFrame()

print(f"Original distribution:")
print(f"Positive samples: {len(positive_df)}")
print(f"Negative samples: {len(negative_df) if not negative_df.empty else 'N/A'}")

# Calculate how many augmentations needed per positive image
if not negative_df.empty:
    target_positive = len(negative_df)
    n_augmentations = int(np.ceil(target_positive / len(positive_df))) - 1
else:
    # If no negative samples in CSV, assume ratio from description
    target_positive = 3748
    n_augmentations = int(np.ceil(target_positive / len(positive_df))) - 1

print(f"Target positive samples: {target_positive}")
print(f"Augmentations per image: {n_augmentations}")

def apply_minimal_augmentation(image_array, aug_type):
    """Apply minimal augmentations suitable for CXR images"""
    if aug_type == 0:
        # Small horizontal shift (±2%)
        shift_x = np.random.uniform(-0.02, 0.02) * image_array.shape[1]
        return shift(image_array, [0, shift_x], mode='nearest')
    elif aug_type == 1:
        # Small vertical shift (±2%)
        shift_y = np.random.uniform(-0.02, 0.02) * image_array.shape[0]
        return shift(image_array, [shift_y, 0], mode='nearest')
    elif aug_type == 2:
        # Very small rotation (±3 degrees)
        angle = np.random.uniform(-3, 3)
        return rotate(image_array, angle, reshape=False, mode='nearest')
    elif aug_type == 3:
        # Slight brightness adjustment
        factor = np.random.uniform(0.95, 1.05)
        return np.clip(image_array * factor, image_array.min(), image_array.max())
    elif aug_type == 4:
        # Horizontal flip (careful with CXR, but acceptable)
        return np.fliplr(image_array)
    else:
        # Combination: small shift + small rotation
        shift_x = np.random.uniform(-0.01, 0.01) * image_array.shape[1]
        shifted = shift(image_array, [0, shift_x], mode='nearest')
        angle = np.random.uniform(-2, 2)
        return rotate(shifted, angle, reshape=False, mode='nearest')

def load_mha_image(filepath):
    """Load .mha image using SimpleITK"""
    image = sitk.ReadImage(filepath)
    array = sitk.GetArrayFromImage(image)
    return array, image

def save_mha_image(array, reference_image, filepath):
    """Save array as .mha using SimpleITK with reference metadata"""
    output_image = sitk.GetImageFromArray(array)
    output_image.CopyInformation(reference_image)
    sitk.WriteImage(output_image, filepath)

# Copy all original images and create new metadata list
new_metadata_rows = []

# Copy negative samples (if any in CSV)
if not negative_df.empty:
    print("\nCopying negative samples...")
    for idx, row in negative_df.iterrows():
        src = os.path.join(images_path, row['img_name'])
        dst = os.path.join(output_images_path, row['img_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
            new_metadata_rows.append(row.to_dict())

# Copy and augment positive samples
print("\nProcessing positive samples...")
augmented_count = 0

for idx, row in positive_df.iterrows():
    img_path = os.path.join(images_path, row['img_name'])
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, skipping...")
        continue
    
    # Copy original
    dst = os.path.join(output_images_path, row['img_name'])
    shutil.copy2(img_path, dst)
    new_metadata_rows.append(row.to_dict())
    
    # Load image
    img_array, reference_img = load_mha_image(img_path)
    
    # Create augmentations
    base_name = Path(row['img_name']).stem
    ext = Path(row['img_name']).suffix
    
    for aug_idx in range(n_augmentations):
        aug_type = aug_idx % 6  # Cycle through augmentation types
        augmented_array = apply_minimal_augmentation(img_array, aug_type)
        
        # Save augmented image
        aug_name = f"{base_name}_aug{aug_idx}{ext}"
        aug_path = os.path.join(output_images_path, aug_name)
        save_mha_image(augmented_array, reference_img, aug_path)
        
        # Create new metadata row
        new_row = row.to_dict()
        new_row['img_name'] = aug_name
        
        # Adjust bounding box coordinates for horizontal flip
        if aug_type == 4:
            new_row['x'] = img_array.shape[1] - row['x'] - row['width']
        
        new_metadata_rows.append(new_row)
        augmented_count += 1
    
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(positive_df)} positive images...")

# Create new metadata CSV
new_df = pd.DataFrame(new_metadata_rows)
new_df = new_df.reset_index(drop=True)
new_df.insert(0, '#', range(len(new_df)))
new_df.to_csv(output_csv_path, index=False)

print(f"\n{'='*50}")
print(f"Upsampling complete!")
print(f"{'='*50}")
print(f"Total images created: {len(new_df)}")
print(f"Positive samples: {len(new_df[new_df['label'] == 1])}")
print(f"Negative samples: {len(new_df[new_df['label'] == 0]) if 0 in new_df['label'].values else 'N/A'}")
print(f"Augmented images created: {augmented_count}")
print(f"\nOutput saved to:")
print(f"Images: {output_images_path}")
print(f"CSV: {output_csv_path}")