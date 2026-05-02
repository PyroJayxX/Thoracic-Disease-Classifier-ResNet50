import pandas as pd
import numpy as np
import os
from pathlib import Path
import shutil
from scipy.ndimage import rotate, shift
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

# Paths
base_path = "./dataset_nodule21/cxr_images/proccessed_data"
images_path = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "metadata.csv")

# New output paths
output_base = os.path.join(base_path, "split_data")  # Changed folder name
train_images_path = os.path.join(output_base, "train", "images")
val_images_path = os.path.join(output_base, "val", "images")
test_images_path = os.path.join(output_base, "test", "images")

train_csv_path = os.path.join(output_base, "train", "metadata_train.csv")
val_csv_path = os.path.join(output_base, "val", "metadata_val.csv")
test_csv_path = os.path.join(output_base, "test", "metadata_test.csv")

# Create output directories
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)

# Read metadata
df = pd.read_csv(csv_path)

print(f"Original dataset:")
print(f"Total rows: {len(df)}")
print(f"Unique images: {df['img_name'].nunique()}")
print(f"Class distribution:\n{df['label'].value_counts()}")

# CRITICAL FIX: Group by image name and get one label per image
# (assuming if image has ANY positive nodule, it's positive)
image_labels = df.groupby('img_name')['label'].max().reset_index()

print(f"\nAfter grouping by image:")
print(f"Unique images: {len(image_labels)}")
print(f"Image-level class distribution:\n{image_labels['label'].value_counts()}")

# STEP 1: SPLIT BY UNIQUE IMAGES (not by rows!)
print(f"\n{'='*50}")
print("STEP 1: Splitting by unique images...")
print(f"{'='*50}")

train_images, temp_images = train_test_split(
    image_labels['img_name'], 
    test_size=0.3, 
    stratify=image_labels['label'], 
    random_state=42
)

# Get labels for temp split
temp_labels = image_labels[image_labels['img_name'].isin(temp_images)]

val_images, test_images = train_test_split(
    temp_images, 
    test_size=0.5, 
    stratify=temp_labels['label'], 
    random_state=42
)

# Now split the full dataframe based on image splits
train_df = df[df['img_name'].isin(train_images)].copy()
val_df = df[df['img_name'].isin(val_images)].copy()
test_df = df[df['img_name'].isin(test_images)].copy()

print(f"\nSplit sizes:")
print(f"Train: {len(train_df)} rows, {train_df['img_name'].nunique()} unique images")
print(f"Val: {len(val_df)} rows, {val_df['img_name'].nunique()} unique images")
print(f"Test: {len(test_df)} rows, {test_df['img_name'].nunique()} unique images")

print(f"\nTrain distribution:\n{train_df['label'].value_counts()}")
print(f"Val distribution:\n{val_df['label'].value_counts()}")
print(f"Test distribution:\n{test_df['label'].value_counts()}")

# STEP 2: AUGMENT ONLY TRAINING SET
print(f"\n{'='*50}")
print("STEP 2: Augmenting training set...")
print(f"{'='*50}")

def apply_minimal_augmentation(image_array, aug_type):
    """Apply minimal augmentations suitable for CXR images"""
    if aug_type == 0:
        shift_x = np.random.uniform(-0.02, 0.02) * image_array.shape[1]
        return shift(image_array, [0, shift_x], mode='nearest')
    elif aug_type == 1:
        shift_y = np.random.uniform(-0.02, 0.02) * image_array.shape[0]
        return shift(image_array, [shift_y, 0], mode='nearest')
    elif aug_type == 2:
        angle = np.random.uniform(-3, 3)
        return rotate(image_array, angle, reshape=False, mode='nearest')
    elif aug_type == 3:
        factor = np.random.uniform(0.95, 1.05)
        return np.clip(image_array * factor, image_array.min(), image_array.max())
    elif aug_type == 4:
        return np.fliplr(image_array)
    else:
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

# Get unique positive and negative IMAGES in training set
train_positive_images = train_df[train_df['label'] == 1]['img_name'].unique()
train_negative_images = train_df[train_df['label'] == 0]['img_name'].unique()

# Get TOTAL ROWS (accounting for multiple nodules per image)
train_positive_rows = len(train_df[train_df['label'] == 1])
train_negative_rows = len(train_df[train_df['label'] == 0])

print(f"\nTraining set before augmentation:")
print(f"Positive images: {len(train_positive_images)}")
print(f"Positive rows: {train_positive_rows}")
print(f"Negative images: {len(train_negative_images)}")
print(f"Negative rows: {train_negative_rows}")

# Calculate augmentations needed based on ROWS, not images
# We want total positive rows ≈ total negative rows
# Current positive rows = train_positive_rows
# After augmentation: train_positive_rows * (1 + n_augmentations) ≈ train_negative_rows
# So: n_augmentations = (train_negative_rows / train_positive_rows) - 1

n_augmentations = int(np.ceil(train_negative_rows / train_positive_rows)) - 1
n_augmentations = max(0, n_augmentations)  # Don't allow negative

print(f"Target: Balance {train_positive_rows} positive rows with {train_negative_rows} negative rows")
print(f"Augmentations per positive IMAGE: {n_augmentations}")
print(f"Expected positive rows after augmentation: ~{train_positive_rows * (1 + n_augmentations)}")

print(f"Augmentations per positive image: {n_augmentations}")

# Process training set
train_metadata_rows = []
augmented_count = 0

# Copy negative samples
print("\nCopying training negative samples...")
train_negative_df = train_df[train_df['label'] == 0]
for idx, row in train_negative_df.iterrows():
    src = os.path.join(images_path, row['img_name'])
    dst = os.path.join(train_images_path, row['img_name'])
    if os.path.exists(src) and not os.path.exists(dst):  # Only copy once
        shutil.copy2(src, dst)
    train_metadata_rows.append(row.to_dict())

# Copy and augment positive samples (by unique image)
print("Processing training positive samples...")
train_positive_df = train_df[train_df['label'] == 1]

processed_images = set()
for img_name in train_positive_images:
    if img_name in processed_images:
        continue
    processed_images.add(img_name)
    
    img_path = os.path.join(images_path, img_name)
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, skipping...")
        continue
    
    # Copy original
    dst = os.path.join(train_images_path, img_name)
    shutil.copy2(img_path, dst)
    
    # Add ALL rows for this image (multiple bounding boxes)
    img_rows = train_positive_df[train_positive_df['img_name'] == img_name]
    for _, row in img_rows.iterrows():
        train_metadata_rows.append(row.to_dict())
    
    # Load image for augmentation
    img_array, reference_img = load_mha_image(img_path)
    
    # Create augmentations
    base_name = Path(img_name).stem
    ext = Path(img_name).suffix
    
    for aug_idx in range(n_augmentations):
        aug_type = aug_idx % 6
        augmented_array = apply_minimal_augmentation(img_array, aug_type)
        
        # Save augmented image
        aug_name = f"{base_name}_aug{aug_idx}{ext}"
        aug_path = os.path.join(train_images_path, aug_name)
        save_mha_image(augmented_array, reference_img, aug_path)
        
        # Create metadata rows for ALL bounding boxes in this image
        for _, row in img_rows.iterrows():
            new_row = row.to_dict()
            new_row['img_name'] = aug_name
            
            # Adjust bounding box for horizontal flip
            if aug_type == 4:
                new_row['x'] = img_array.shape[1] - row['x'] - row['width']
            
            train_metadata_rows.append(new_row)
        
        augmented_count += 1
    
    if len(processed_images) % 100 == 0:
        print(f"Processed {len(processed_images)}/{len(train_positive_images)} positive images...")

# Save training metadata
train_final_df = pd.DataFrame(train_metadata_rows)
train_final_df = train_final_df.reset_index(drop=True)
train_final_df.insert(0, '#', range(len(train_final_df)))
train_final_df.to_csv(train_csv_path, index=False)

print(f"\n✓ Training set complete!")
print(f"Total training rows: {len(train_final_df)}")
print(f"Unique images: {train_final_df['img_name'].nunique()}")
print(f"Positive rows: {len(train_final_df[train_final_df['label'] == 1])}")
print(f"Negative rows: {len(train_final_df[train_final_df['label'] == 0])}")

# STEP 3: COPY VAL AND TEST (NO AUGMENTATION)
print(f"\n{'='*50}")
print("STEP 3: Copying validation and test sets...")
print(f"{'='*50}")

# Copy validation set
print("\nCopying validation samples...")
copied_val = set()
for idx, row in val_df.iterrows():
    src = os.path.join(images_path, row['img_name'])
    dst = os.path.join(val_images_path, row['img_name'])
    if os.path.exists(src) and row['img_name'] not in copied_val:
        shutil.copy2(src, dst)
        copied_val.add(row['img_name'])

val_df_final = val_df.copy()
val_df_final = val_df_final.reset_index(drop=True)
val_df_final.insert(0, '#', range(len(val_df_final)))
val_df_final.to_csv(val_csv_path, index=False)

print(f"✓ Validation set: {len(val_df_final)} rows, {val_df_final['img_name'].nunique()} unique images")

# Copy test set
print("\nCopying test samples...")
copied_test = set()
for idx, row in test_df.iterrows():
    src = os.path.join(images_path, row['img_name'])
    dst = os.path.join(test_images_path, row['img_name'])
    if os.path.exists(src) and row['img_name'] not in copied_test:
        shutil.copy2(src, dst)
        copied_test.add(row['img_name'])

test_df_final = test_df.copy()
test_df_final = test_df_final.reset_index(drop=True)
test_df_final.insert(0, '#', range(len(test_df_final)))
test_df_final.to_csv(test_csv_path, index=False)

print(f"✓ Test set: {len(test_df_final)} rows, {test_df_final['img_name'].nunique()} unique images")

# FINAL VERIFICATION
print(f"\n{'='*60}")
print("OVERLAP VERIFICATION:")
print(f"{'='*60}")

train_unique = set(train_final_df['img_name'].str.replace(r'_aug\d+', '', regex=True))
val_unique = set(val_df_final['img_name'])
test_unique = set(test_df_final['img_name'])

overlap_train_val = train_unique & val_unique
overlap_train_test = train_unique & test_unique
overlap_val_test = val_unique & test_unique

print(f"Train base images: {len(train_unique)}")
print(f"Val images: {len(val_unique)}")
print(f"Test images: {len(test_unique)}")
print(f"\nTrain-Val overlap: {len(overlap_train_val)} (should be 0)")
print(f"Train-Test overlap: {len(overlap_train_test)} (should be 0)")
print(f"Val-Test overlap: {len(overlap_val_test)} (should be 0)")

if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
    print("\n✓✓✓ SUCCESS! No data leakage detected!")
else:
    print("\n⚠️ WARNING: Still has overlap!")
    if len(overlap_train_val) > 0:
        print(f"Train-Val overlap examples: {list(overlap_train_val)[:5]}")

print(f"\n{'='*60}")
print("FINAL SUMMARY:")
print(f"{'='*60}")
print(f"Output folder: {output_base}")