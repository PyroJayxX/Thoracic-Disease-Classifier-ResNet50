import os
import shutil
import pandas as pd
import numpy as np

# =========================
# Configuration
# =========================

# Source directory containing images and CSV
SOURCE_IMAGE_DIR = './dataset_unbalanced/images'
SOURCE_CSV_PATH = './dataset_unbalanced/Data_Entry_2017.csv'

# Destination directory for the BALANCED dataset
DEST_DATASET_DIR = './dataset_pneumothorax'
DEST_IMAGE_DIR = os.path.join(DEST_DATASET_DIR, 'images')
DEST_CSV_PATH = os.path.join(DEST_DATASET_DIR, 'balanced_dataset.csv')

# Random seed for reproducibility
RANDOM_SEED = 42

# =========================
# Step 1: Create directories
# =========================

try:
    os.makedirs(DEST_IMAGE_DIR, exist_ok=True)
    print(f"✓ Created directory: {DEST_IMAGE_DIR}")
except Exception as e:
    print(f"✗ Error creating directory {DEST_IMAGE_DIR}: {e}")
    exit(1)

# =========================
# Step 2: Load and analyze dataset
# =========================

try:
    df = pd.read_csv(SOURCE_CSV_PATH)
    print(f"✓ Loaded CSV with {len(df):,} total images")
except Exception as e:
    print(f"✗ Error reading CSV {SOURCE_CSV_PATH}: {e}")
    exit(1)

# Check if 'Finding Labels' column exists (standard NIH ChestX-ray14 format)
if 'Finding Labels' in df.columns:
    print("\n=== Dataset Format: NIH ChestX-ray14 ===")
    
    # Create binary Pneumothorax column from 'Finding Labels'
    df['Pneumothorax'] = df['Finding Labels'].apply(
        lambda x: 1 if 'Pneumothorax' in str(x) else 0
    )
    
    # Identify image path column
    if 'Image Index' in df.columns:
        path_column = 'Image Index'
    elif 'Path' in df.columns:
        path_column = 'Path'
    else:
        path_column = df.columns[0]
        print(f"⚠ Using first column as image path: '{path_column}'")
    
elif 'Pneumothorax' in df.columns:
    print("\n=== Dataset already has binary columns ===")
    path_column = 'Path' if 'Path' in df.columns else df.columns[0]
else:
    print("✗ Could not identify Pneumothorax labels in CSV")
    print("Available columns:", df.columns.tolist())
    exit(1)

# Count class distribution
n_positive = df['Pneumothorax'].sum()
n_negative = len(df) - n_positive

print(f"\n=== Original Distribution ===")
print(f"Pneumothorax (Positive):     {n_positive:,} ({n_positive/len(df)*100:.2f}%)")
print(f"No Pneumothorax (Negative):  {n_negative:,} ({n_negative/len(df)*100:.2f}%)")
print(f"Imbalance ratio:             1:{n_negative/n_positive:.1f}")

# =========================
# Step 3: Create balanced dataset
# =========================

print(f"\n=== Creating Balanced Dataset ===")

# Get ALL positive samples (Pneumothorax)
df_positive = df[df['Pneumothorax'] == 1].copy()
n_pneumothorax = len(df_positive)

print(f"✓ Using ALL {n_pneumothorax:,} Pneumothorax cases")

# Get negative samples (No Pneumothorax)
df_negative = df[df['Pneumothorax'] == 0].copy()

# Sample EQUAL number of negative cases to match positive
TARGET_NEGATIVE_SAMPLES = n_pneumothorax

if len(df_negative) >= TARGET_NEGATIVE_SAMPLES:
    df_negative = df_negative.sample(n=TARGET_NEGATIVE_SAMPLES, random_state=RANDOM_SEED)
    print(f"✓ Sampled {TARGET_NEGATIVE_SAMPLES:,} negative cases (matched to positive)")
else:
    print(f"⚠ Warning: Only {len(df_negative):,} negative cases available (need {TARGET_NEGATIVE_SAMPLES:,})")

# Combine balanced datasets
df_balanced = pd.concat([df_positive, df_negative], ignore_index=True)

# Shuffle the balanced dataset
df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\n=== Balanced Dataset Summary ===")
print(f"Total samples:     {len(df_balanced):,}")
print(f"Positive:          {df_balanced['Pneumothorax'].sum():,} ({df_balanced['Pneumothorax'].sum()/len(df_balanced)*100:.1f}%)")
print(f"Negative:          {len(df_balanced) - df_balanced['Pneumothorax'].sum():,} ({(len(df_balanced) - df_balanced['Pneumothorax'].sum())/len(df_balanced)*100:.1f}%)")
print(f"Balance ratio:     1:1 (perfect balance)")

# =========================
# Step 4: Copy selected images
# =========================

print(f"\n=== Copying Images ===")

# Get list of image filenames to copy
images_to_copy = df_balanced[path_column].apply(lambda x: os.path.basename(x)).tolist()

copied_count = 0
failed_count = 0
failed_images = []

for idx, img_file in enumerate(images_to_copy, 1):
    src_path = os.path.join(SOURCE_IMAGE_DIR, img_file)
    dest_path = os.path.join(DEST_IMAGE_DIR, img_file)
    
    if not os.path.exists(src_path):
        print(f"⚠ Image not found: {img_file}")
        failed_count += 1
        failed_images.append(img_file)
        continue
    
    try:
        shutil.copy2(src_path, dest_path)
        copied_count += 1
        
        # Progress indicator every 500 images
        if copied_count % 500 == 0:
            print(f"  Progress: {copied_count:,}/{len(images_to_copy):,} images ({copied_count/len(images_to_copy)*100:.1f}%)")
            
    except Exception as e:
        print(f"✗ Error copying {img_file}: {e}")
        failed_count += 1
        failed_images.append(img_file)

print(f"\n✓ Successfully copied {copied_count:,} images")
if failed_count > 0:
    print(f"⚠ Failed to copy {failed_count:,} images")
    print(f"  First 5 failed: {failed_images[:5]}")

# Remove failed images from dataframe
if failed_images:
    df_balanced = df_balanced[~df_balanced[path_column].apply(
        lambda x: os.path.basename(x) in failed_images
    )].copy()
    print(f"✓ Removed {len(failed_images):,} failed images from CSV")

# =========================
# Step 5: Generate balanced CSV
# =========================

print(f"\n=== Generating CSV ===")

# Update paths to reflect new location
df_balanced['Path'] = df_balanced[path_column].apply(
    lambda x: os.path.join('images', os.path.basename(x))
)

# Create clean CSV with essential columns
if 'Finding Labels' in df_balanced.columns:
    # Keep original finding labels for reference
    df_output = df_balanced[[path_column, 'Finding Labels', 'Pneumothorax']].copy()
    df_output.columns = ['Image_Index', 'Finding_Labels', 'Pneumothorax']
    
    # Update Image_Index to match new path
    df_output['Path'] = df_balanced['Path']
else:
    # Use existing structure
    df_output = df_balanced.copy()

# Add binary label column for convenience
df_output['Binary_Label'] = df_output['Pneumothorax'].astype(int)

try:
    df_output.to_csv(DEST_CSV_PATH, index=False)
    print(f"✓ CSV saved to {DEST_CSV_PATH}")
    print(f"  Contains {len(df_output):,} entries")
except Exception as e:
    print(f"✗ Error saving CSV: {e}")
    exit(1)

# =========================
# Step 6: Verification
# =========================

print(f"\n=== Dataset Verification ===")

# Count actual copied files
actual_images = len([f for f in os.listdir(DEST_IMAGE_DIR) if f.lower().endswith('.png')])
expected_images = len(df_balanced)

print(f"Expected images: {expected_images:,}")
print(f"Actual images:   {actual_images:,}")

if actual_images == expected_images:
    print(f"✓ All images copied successfully!")
else:
    print(f"⚠ Mismatch: {abs(actual_images - expected_images):,} image(s) difference")

# Verify class balance
positive_count = df_balanced['Pneumothorax'].sum()
negative_count = len(df_balanced) - positive_count

print(f"\nFinal Class Distribution:")
print(f"  Pneumothorax:     {positive_count:,} ({positive_count/len(df_balanced)*100:.1f}%)")
print(f"  No Pneumothorax:  {negative_count:,} ({negative_count/len(df_balanced)*100:.1f}%)")

# =========================
# Summary
# =========================

print(f"\n{'='*70}")
print(f"BALANCED PNEUMOTHORAX DATASET CREATION COMPLETE")
print(f"{'='*70}")
print(f"Location:     {DEST_DATASET_DIR}")
print(f"Images:       {DEST_IMAGE_DIR} ({actual_images:,} files)")
print(f"CSV:          {DEST_CSV_PATH}")
print(f"\nDataset Characteristics:")
print(f"  Total samples:        {len(df_balanced):,}")
print(f"  Pneumothorax:         {positive_count:,} (50.0%)")
print(f"  No Pneumothorax:      {negative_count:,} (50.0%)")
print(f"  Balance ratio:        1:1 (Perfect)")
print(f"\nReady for binary classification training!")
print(f"{'='*70}")