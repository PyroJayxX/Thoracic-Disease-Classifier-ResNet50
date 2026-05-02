import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import random
from scipy.ndimage import rotate, shift
import SimpleITK as sitk
import shutil

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
IMAGE_DIR = "./dataset_nodule/images/images"
CSV_PATH = "./dataset_nodule/jsrt_metadata.csv"
OUTPUT_DIR = "./dataset_nodule_augmented"
TEST_RATIO = 0.20  # Only split out test set, keep originals for CV

# Delete old directory if it exists
if os.path.exists(OUTPUT_DIR):
    print(f"Removing old directory: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)
    print("  ✓ Directory removed")

# Create output directories
for split in ['trainval', 'test']:
    for class_name in ['malignant', 'benign', 'non-nodule']:
        Path(f"{OUTPUT_DIR}/{split}/{class_name}").mkdir(parents=True, exist_ok=True)

def augment_image(img_array, aug_type):
    """Apply specific augmentation type using scipy - minimal medical-appropriate augmentations"""
    if aug_type == 'h_translation':
        # Horizontal translation: ±2% of width
        shift_pct = np.random.uniform(-0.02, 0.02)
        shift_pixels = shift_pct * img_array.shape[1]
        return shift(img_array, shift=[0, shift_pixels], mode='nearest')
    
    elif aug_type == 'v_translation':
        # Vertical translation: ±2% of height
        shift_pct = np.random.uniform(-0.02, 0.02)
        shift_pixels = shift_pct * img_array.shape[0]
        return shift(img_array, shift=[shift_pixels, 0], mode='nearest')
    
    elif aug_type == 'rotation':
        # Rotation: ±3 degrees
        angle = np.random.uniform(-3, 3)
        return rotate(img_array, angle, reshape=False, mode='nearest')
    
    elif aug_type == 'brightness':
        # Brightness adjustment: 0.95-1.05, clipped to original range
        factor = np.random.uniform(0.95, 1.05)
        return np.clip(img_array * factor, img_array.min(), img_array.max())
    
    elif aug_type == 'h_flip':
        # Horizontal flip
        return np.fliplr(img_array)
    
    elif aug_type == 'combined':
        # Combined: horizontal shift ±1% + rotation ±2°
        h_shift_pct = np.random.uniform(-0.01, 0.01)
        h_shift_pixels = h_shift_pct * img_array.shape[1]
        shifted = shift(img_array, shift=[0, h_shift_pixels], mode='nearest')
        angle = np.random.uniform(-2, 2)
        return rotate(shifted, angle, reshape=False, mode='nearest')
    
    return img_array

def load_image(filepath):
    """Load image - supports both standard formats and .mha"""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.mha':
        # Load using SimpleITK for .mha files
        sitk_img = sitk.ReadImage(filepath)
        img_array = sitk.GetArrayFromImage(sitk_img)
        # For 3D volumes, take first slice
        if len(img_array.shape) == 3:
            img_array = img_array[0]
        return img_array, sitk_img
    else:
        # Load using PIL for standard formats (png, jpg, etc.)
        img = Image.open(filepath).convert('L')
        img_array = np.array(img)
        return img_array, None

def save_image(img_array, filepath, reference_sitk=None):
    """Save image - handles both standard formats and .mha"""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.mha' and reference_sitk is not None:
        # Save as .mha using SimpleITK
        output_img = sitk.GetImageFromArray(img_array)
        output_img.CopyInformation(reference_sitk)
        sitk.WriteImage(output_img, filepath)
    else:
        # Save as standard image format
        # Ensure values are in valid range
        if img_array.dtype != np.uint8:
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_array = np.zeros_like(img_array, dtype=np.uint8)
        
        img = Image.fromarray(img_array)
        img.save(filepath)

def load_all_images(df_class, class_name, image_dir):
    """Load all original images for a class"""
    images = []
    
    print(f"  Loading {class_name} images...")
    for idx, row in df_class.iterrows():
        img_path = os.path.join(image_dir, row['study_id'])
        if os.path.exists(img_path):
            try:
                img_array, reference_sitk = load_image(img_path)
                # Store metadata with image
                metadata = row.to_dict()
                images.append((img_array, row['study_id'], metadata, reference_sitk))
            except Exception as e:
                print(f"    Warning: Could not load {row['study_id']}: {e}")
        else:
            print(f"    Warning: {img_path} not found")
    
    print(f"    ✓ {len(images)} images loaded")
    return images

def split_class_images(images, class_name):
    """Split images into trainval (for CV - ORIGINAL ONLY) and test sets"""
    random.shuffle(images)
    
    total = len(images)
    test_size = int(total * TEST_RATIO)
    
    trainval_imgs = images[:-test_size] if test_size > 0 else images
    test_imgs = images[-test_size:] if test_size > 0 else []
    
    return trainval_imgs, test_imgs

def augment_train_set(trainval_imgs, class_name, target_samples):
    """Augment training+validation set to reach target number of samples"""
    original_count = len(trainval_imgs)
    num_augmentations_needed = target_samples - original_count
    
    if num_augmentations_needed <= 0:
        print(f"    No augmentation needed (already have {original_count} samples)")
        return trainval_imgs
    
    print(f"    Augmenting from {original_count} to {target_samples} samples...")
    
    aug_types = ['h_translation', 'v_translation', 'rotation', 
                 'brightness', 'h_flip', 'combined']
    
    augmented_imgs = trainval_imgs.copy()
    
    for i in range(num_augmentations_needed):
        # Select random original training image
        img_array, filename, metadata, reference_sitk = random.choice(trainval_imgs)
        
        # Apply random augmentation
        aug_type = random.choice(aug_types)
        aug_img = augment_image(img_array, aug_type)
        
        # Create augmented filename - preserve extension
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        aug_name = f"{base_name}_aug_{i}_{aug_type}{ext}"
        
        # Copy metadata and update
        aug_metadata = metadata.copy()
        aug_metadata['study_id'] = aug_name
        aug_metadata['augmentation'] = aug_type
        aug_metadata['original_image'] = filename
        
        augmented_imgs.append((aug_img, aug_name, aug_metadata, reference_sitk))
        
        if (i + 1) % 200 == 0:
            print(f"      Generated {i + 1}/{num_augmentations_needed} augmentations...")
    
    print(f"    ✓ Augmentation complete: {len(augmented_imgs)} total samples")
    return augmented_imgs

def save_split(images, split_name, class_name, apply_original_flag=False):
    """Save images for a specific split and collect metadata"""
    metadata_list = []
    
    for img_array, filename, metadata, reference_sitk in images:
        output_path = f"{OUTPUT_DIR}/{split_name}/{class_name}/{filename}"
        
        try:
            save_image(img_array, output_path, reference_sitk)
            
            # Add split and path info to metadata
            metadata_entry = metadata.copy()
            metadata_entry['split'] = split_name
            metadata_entry['class'] = class_name
            metadata_entry['filepath'] = f"{split_name}/{class_name}/{filename}"
            
            # Mark as original if not augmented
            if apply_original_flag and 'augmentation' not in metadata_entry:
                metadata_entry['augmentation'] = 'original'
                metadata_entry['original_image'] = filename
            
            metadata_list.append(metadata_entry)
        except Exception as e:
            print(f"    Warning: Could not save {filename}: {e}")
    
    return metadata_list

# Main execution
print("="*70)
print("DATASET SPLIT & AUGMENTATION (TRAIN ONLY)")
print("="*70)

print(f"\nLoading metadata from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"  ✓ Loaded {len(df)} records")

# Separate classes based on the 'state' column
df_malignant = df[df['state'] == 'malignant']
df_benign = df[df['state'] == 'benign']
df_non_nodule = df[df['state'] == 'non-nodule']

print("\n" + "="*70)
print("ORIGINAL CLASS DISTRIBUTION")
print("="*70)
print(f"Malignant:   {len(df_malignant):>4} samples")
print(f"Benign:      {len(df_benign):>4} samples")
print(f"Non-nodule:  {len(df_non_nodule):>4} samples")
print(f"Total:       {len(df):>4} samples")

# Load all images
print("\n" + "="*70)
print("LOADING IMAGES")
print("="*70)

malignant_images = load_all_images(df_malignant, 'malignant', IMAGE_DIR)
benign_images = load_all_images(df_benign, 'benign', IMAGE_DIR)
non_nodule_images = load_all_images(df_non_nodule, 'non-nodule', IMAGE_DIR)

# Split each class
print("\n" + "="*70)
print("SPLITTING DATASET")
print("="*70)
print(f"Split ratio: TrainVal (ORIGINAL for CV)={1-TEST_RATIO:.0%}, Test={TEST_RATIO:.0%}")

print("\nMalignant:")
mal_trainval, mal_test = split_class_images(malignant_images, 'malignant')
print(f"  TrainVal: {len(mal_trainval)} | Test: {len(mal_test)}")

print("\nBenign:")
ben_trainval, ben_test = split_class_images(benign_images, 'benign')
print(f"  TrainVal: {len(ben_trainval)} | Test: {len(ben_test)}")

print("\nNon-nodule:")
non_trainval, non_test = split_class_images(non_nodule_images, 'non-nodule')
print(f"  TrainVal: {len(non_trainval)} | Test: {len(non_test)}")

# Save all splits - NO AUGMENTATION HERE
print("\n" + "="*70)
print("SAVING DATASET")
print("="*70)
print("NOTE: Saving ORIGINAL images only - augmentation will be done during CV training")

print("\nSaving trainval set (original images for cross-validation)...")
mal_trainval_meta = save_split(mal_trainval, 'trainval', 'malignant', apply_original_flag=True)
ben_trainval_meta = save_split(ben_trainval, 'trainval', 'benign', apply_original_flag=True)
non_trainval_meta = save_split(non_trainval, 'trainval', 'non-nodule', apply_original_flag=True)
print(f"  ✓ TrainVal: {len(mal_trainval) + len(ben_trainval) + len(non_trainval)} original images saved")

print("\nSaving test set...")
mal_test_meta = save_split(mal_test, 'test', 'malignant', apply_original_flag=True)
ben_test_meta = save_split(ben_test, 'test', 'benign', apply_original_flag=True)
non_test_meta = save_split(non_test, 'test', 'non-nodule', apply_original_flag=True)
print(f"  ✓ Test: {len(mal_test) + len(ben_test) + len(non_test)} original images saved")

# Combine metadata for each split
print("\n" + "="*70)
print("CREATING METADATA DATAFRAMES")
print("="*70)

trainval_df = pd.DataFrame(mal_trainval_meta + ben_trainval_meta + non_trainval_meta)
print(f"  Train+Val DataFrame: {len(trainval_df)} rows, {len(trainval_df.columns)} columns")

test_df = pd.DataFrame(mal_test_meta + ben_test_meta + non_test_meta)
print(f"  Test DataFrame: {len(test_df)} rows, {len(test_df.columns)} columns")

# Save metadata CSVs
trainval_csv = f"{OUTPUT_DIR}/trainval_metadata.csv"
test_csv = f"{OUTPUT_DIR}/test_metadata.csv"
combined_csv = f"{OUTPUT_DIR}/combined_metadata.csv"

print("\nSaving CSV files...")
try:
    trainval_df.to_csv(trainval_csv, index=False)
    print(f"  ✓ Saved: {trainval_csv}")
except Exception as e:
    print(f"  ✗ Error saving train+val CSV: {e}")

try:
    test_df.to_csv(test_csv, index=False)
    print(f"  ✓ Saved: {test_csv}")
except Exception as e:
    print(f"  ✗ Error saving test CSV: {e}")

# Also save a combined CSV
try:
    combined_df = pd.concat([trainval_df, test_df], ignore_index=True)
    combined_df.to_csv(combined_csv, index=False)
    print(f"  ✓ Saved: {combined_csv}")
except Exception as e:
    print(f"  ✗ Error saving combined CSV: {e}")

# Verify files exist
print("\nVerifying CSV files...")
for csv_file in [trainval_csv, test_csv, combined_csv]:
    if os.path.exists(csv_file):
        size = os.path.getsize(csv_file)
        print(f"  ✓ {csv_file} exists ({size:,} bytes)")
    else:
        print(f"  ✗ {csv_file} NOT FOUND!")

print("\n" + "="*70)
print("METADATA CSV FILES SAVED")
print("="*70)
print(f"TrainVal: {trainval_csv} ({len(trainval_df):>4} rows)")
print(f"Test:     {test_csv} ({len(test_df):>4} rows)")
print(f"Combined: {combined_csv} ({len(combined_df):>4} rows)")

# Print final statistics
print("\n" + "="*70)
print("FINAL DATASET STATISTICS")
print("="*70)

print(f"\nMalignant:")
print(f"  TrainVal: {len(mal_trainval):>4} | Test: {len(mal_test):>4}")

print(f"\nBenign:")
print(f"  TrainVal: {len(ben_trainval):>4} | Test: {len(ben_test):>4}")

print(f"\nNon-nodule:")
print(f"  TrainVal: {len(non_trainval):>4} | Test: {len(non_test):>4}")

total_trainval = len(mal_trainval) + len(ben_trainval) + len(non_trainval)
total_test = len(mal_test) + len(ben_test) + len(non_test)
grand_total = total_trainval + total_test

print(f"\n{'─'*70}")
print(f"TOTALS:")
print(f"  TrainVal: {total_trainval:>4} | Test: {total_test:>4} | Grand Total: {grand_total:>4}")

print("\n" + "="*70)
print("DATASET STRUCTURE")
print("="*70)
print(f"{OUTPUT_DIR}/")
print("  ├── trainval/        (ORIGINAL images only for 5-fold CV)")
print("  │   ├── malignant/")
print("  │   ├── benign/")
print("  │   └── non-nodule/")
print("  ├── test/            (held-out test set)")
print("  │   ├── malignant/")
print("  │   ├── benign/")
print("  │   └── non-nodule/")
print("  ├── trainval_metadata.csv  (original images for CV)")
print("  ├── test_metadata.csv      (held-out test data)")
print("  └── combined_metadata.csv")

print("\n" + "="*70)
print("✓ DATASET PREPARATION COMPLETE!")
print("="*70)
print("\nKey points:")
print("  ✓ Old directory removed")
print("  ✓ Data split: 80% trainval (ORIGINAL), 20% test (ORIGINAL)")
print("  ✓ NO augmentation saved - will be done ON-THE-FLY during CV training")
print("  ✓ Test set contains ONLY original images")
print("  ✓ Each CV fold will augment its training portion independently")
print("  ✓ No data leakage - validation folds see only original images")