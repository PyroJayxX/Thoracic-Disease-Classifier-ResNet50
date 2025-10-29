"""
Combined Dataset Creator for Pneumothorax Detection
Combines dataset_pneumothorax and dataset_pneumothorax_small into a single dataset
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("="*80)
print("COMBINED DATASET CREATOR")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Source datasets
DATASET1_DIR = "./dataset_pneumothorax/"
DATASET1_CSV = os.path.join(DATASET1_DIR, "pneumothorax.csv")
DATASET1_IMAGES = os.path.join(DATASET1_DIR, "images")

DATASET2_DIR = "./dataset_pneumothorax_small/"
DATASET2_CSV = os.path.join(DATASET2_DIR, "train_data.csv")
DATASET2_IMAGES = os.path.join(DATASET2_DIR, "small_train_dataset")

# Combined dataset output
COMBINED_DIR = "./combined_dataset/"
COMBINED_IMAGES_DIR = os.path.join(COMBINED_DIR, "images")
COMBINED_CSV = os.path.join(COMBINED_DIR, "combined_pneumothorax.csv")

# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================

print(f"\n📁 Creating combined dataset directory...")
os.makedirs(COMBINED_DIR, exist_ok=True)
os.makedirs(COMBINED_IMAGES_DIR, exist_ok=True)
print(f"✓ Created: {COMBINED_DIR}")
print(f"✓ Created: {COMBINED_IMAGES_DIR}")

# ============================================================================
# LOAD DATASET 1 (LARGE DATASET)
# ============================================================================

print(f"\n" + "="*80)
print("LOADING DATASET 1: dataset_pneumothorax (LARGE)")
print("="*80)

try:
    df1 = pd.read_csv(DATASET1_CSV)
    print(f"✓ Loaded CSV: {len(df1):,} images")
except Exception as e:
    print(f"✗ Error loading {DATASET1_CSV}: {e}")
    exit(1)

# Process Dataset 1
print(f"\nDataset 1 columns: {df1.columns.tolist()}")

# Standardize column names
if 'Binary_Label' in df1.columns:
    df1['binary_label'] = df1['Binary_Label']
elif 'Pneumothorax' in df1.columns:
    df1['binary_label'] = df1['Pneumothorax'].astype(int)
else:
    print("✗ Could not find label column in Dataset 1")
    exit(1)

# Standardize image filename column
if 'Image_Index' in df1.columns:
    df1['image_filename'] = df1['Image_Index']
else:
    print("✗ Could not find Image_Index column in Dataset 1")
    exit(1)

# Create source paths
df1['source_path'] = df1['image_filename'].apply(lambda x: os.path.join(DATASET1_IMAGES, x))

# Verify images exist
df1['exists'] = df1['source_path'].apply(os.path.exists)
missing_count_1 = (~df1['exists']).sum()

if missing_count_1 > 0:
    print(f"⚠ Warning: {missing_count_1} images not found in Dataset 1")
    df1 = df1[df1['exists']].copy()

df1 = df1.drop(columns=['exists'])

print(f"\n✓ Dataset 1 processed:")
print(f"  Total images: {len(df1):,}")
print(f"  Pneumothorax: {df1['binary_label'].sum():,} ({df1['binary_label'].sum()/len(df1)*100:.1f}%)")
print(f"  Negative: {(len(df1) - df1['binary_label'].sum()):,} ({(len(df1) - df1['binary_label'].sum())/len(df1)*100:.1f}%)")

# ============================================================================
# LOAD DATASET 2 (SMALL DATASET)
# ============================================================================

print(f"\n" + "="*80)
print("LOADING DATASET 2: dataset_pneumothorax_small (SMALL)")
print("="*80)

try:
    df2 = pd.read_csv(DATASET2_CSV)
    print(f"✓ Loaded CSV: {len(df2):,} images")
except Exception as e:
    print(f"✗ Error loading {DATASET2_CSV}: {e}")
    exit(1)

# Process Dataset 2
print(f"\nDataset 2 columns: {df2.columns.tolist()}")

# Standardize column names
if 'Binary_Label' in df2.columns:
    df2['binary_label'] = df2['Binary_Label']
elif 'Pneumothorax' in df2.columns:
    df2['binary_label'] = df2['Pneumothorax'].astype(int)
elif 'Label' in df2.columns:
    df2['binary_label'] = df2['Label'].astype(int)
elif 'target' in df2.columns:
    df2['binary_label'] = df2['target'].astype(int)
else:
    print("✗ Could not find label column in Dataset 2")
    print(f"Available columns: {df2.columns.tolist()}")
    exit(1)

# Standardize image filename column
if 'file_name' in df2.columns:
    df2['image_filename'] = df2['file_name']
elif 'Image_Index' in df2.columns:
    df2['image_filename'] = df2['Image_Index']
else:
    print("✗ Could not find filename column in Dataset 2")
    exit(1)

# Create source paths
df2['source_path'] = df2['image_filename'].apply(lambda x: os.path.join(DATASET2_IMAGES, x))

# Verify images exist
df2['exists'] = df2['source_path'].apply(os.path.exists)
missing_count_2 = (~df2['exists']).sum()

if missing_count_2 > 0:
    print(f"⚠ Warning: {missing_count_2} images not found in Dataset 2")
    df2 = df2[df2['exists']].copy()

df2 = df2.drop(columns=['exists'])

print(f"\n✓ Dataset 2 processed:")
print(f"  Total images: {len(df2):,}")
print(f"  Pneumothorax: {df2['binary_label'].sum():,} ({df2['binary_label'].sum()/len(df2)*100:.1f}%)")
print(f"  Negative: {(len(df2) - df2['binary_label'].sum()):,} ({(len(df2) - df2['binary_label'].sum())/len(df2)*100:.1f}%)")

# ============================================================================
# HANDLE FILENAME CONFLICTS
# ============================================================================

print(f"\n" + "="*80)
print("CHECKING FOR FILENAME CONFLICTS")
print("="*80)

# Check for duplicate filenames
df1_filenames = set(df1['image_filename'].tolist())
df2_filenames = set(df2['image_filename'].tolist())
conflicts = df1_filenames.intersection(df2_filenames)

if len(conflicts) > 0:
    print(f"⚠ Found {len(conflicts)} conflicting filenames")
    print(f"  Renaming Dataset 2 files with 'small_' prefix...")
    
    # Rename conflicting files from Dataset 2
    df2['image_filename'] = df2['image_filename'].apply(
        lambda x: f"small_{x}" if x in conflicts else x
    )
    print(f"✓ Renamed {len(conflicts)} files")
else:
    print(f"✓ No filename conflicts found")

# ============================================================================
# COPY IMAGES TO COMBINED DIRECTORY
# ============================================================================

print(f"\n" + "="*80)
print("COPYING IMAGES TO COMBINED DIRECTORY")
print("="*80)

# Copy Dataset 1 images
print(f"\n📋 Copying {len(df1):,} images from Dataset 1...")
copied_1 = 0
failed_1 = 0

for idx, row in tqdm(df1.iterrows(), total=len(df1), desc="Dataset 1"):
    src = row['source_path']
    dst = os.path.join(COMBINED_IMAGES_DIR, row['image_filename'])
    
    try:
        shutil.copy2(src, dst)
        copied_1 += 1
    except Exception as e:
        print(f"✗ Failed to copy {src}: {e}")
        failed_1 += 1

print(f"✓ Dataset 1: Copied {copied_1:,} images ({failed_1} failed)")

# Copy Dataset 2 images
print(f"\n📋 Copying {len(df2):,} images from Dataset 2...")
copied_2 = 0
failed_2 = 0

for idx, row in tqdm(df2.iterrows(), total=len(df2), desc="Dataset 2"):
    src = row['source_path']
    dst = os.path.join(COMBINED_IMAGES_DIR, row['image_filename'])
    
    try:
        shutil.copy2(src, dst)
        copied_2 += 1
    except Exception as e:
        print(f"✗ Failed to copy {src}: {e}")
        failed_2 += 1

print(f"✓ Dataset 2: Copied {copied_2:,} images ({failed_2} failed)")

# ============================================================================
# CREATE COMBINED CSV
# ============================================================================

print(f"\n" + "="*80)
print("CREATING COMBINED CSV")
print("="*80)

# Select only necessary columns for combined dataset
df1_clean = df1[['image_filename', 'binary_label']].copy()
df2_clean = df2[['image_filename', 'binary_label']].copy()

# Add source column to track origin
df1_clean['source'] = 'dataset_pneumothorax'
df2_clean['source'] = 'dataset_pneumothorax_small'

# Combine dataframes
combined_df = pd.concat([df1_clean, df2_clean], ignore_index=True)

# Rename columns to standard names
combined_df = combined_df.rename(columns={
    'image_filename': 'Image_Index',
    'binary_label': 'Binary_Label'
})

# Add Pneumothorax column for compatibility
combined_df['Pneumothorax'] = combined_df['Binary_Label']

# Save combined CSV
combined_df.to_csv(COMBINED_CSV, index=False)
print(f"✓ Combined CSV saved to: {COMBINED_CSV}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n" + "="*80)
print("COMBINED DATASET SUMMARY")
print("="*80)

print(f"\n📊 Statistics:")
print(f"  Total images: {len(combined_df):,}")
print(f"  From Dataset 1: {len(df1_clean):,} ({len(df1_clean)/len(combined_df)*100:.1f}%)")
print(f"  From Dataset 2: {len(df2_clean):,} ({len(df2_clean)/len(combined_df)*100:.1f}%)")

print(f"\n🏷️  Labels:")
print(f"  Pneumothorax (Positive): {combined_df['Binary_Label'].sum():,} ({combined_df['Binary_Label'].sum()/len(combined_df)*100:.1f}%)")
print(f"  No Pneumothorax (Negative): {(len(combined_df) - combined_df['Binary_Label'].sum()):,} ({(len(combined_df) - combined_df['Binary_Label'].sum())/len(combined_df)*100:.1f}%)")

print(f"\n📁 Output Files:")
print(f"  Directory: {COMBINED_DIR}")
print(f"  CSV: {COMBINED_CSV}")
print(f"  Images: {COMBINED_IMAGES_DIR}")
print(f"  Total image files: {len(os.listdir(COMBINED_IMAGES_DIR)):,}")

# Verify balance
pos_count = combined_df['Binary_Label'].sum()
neg_count = len(combined_df) - pos_count
balance_ratio = pos_count / neg_count if neg_count > 0 else 0

print(f"\n⚖️  Class Balance:")
print(f"  Ratio (Positive:Negative): 1:{1/balance_ratio:.2f}")

if 0.8 <= balance_ratio <= 1.2:
    print(f"  ✅ Dataset is well balanced!")
else:
    print(f"  ⚠️  Dataset may be imbalanced")

print(f"\n" + "="*80)
print("✅ COMBINED DATASET CREATION COMPLETE!")
print("="*80)

print(f"\n📖 To use the combined dataset in your training notebook:")
print(f"""
BASE_DIR = "./combined_dataset/"
CSV_PATH = os.path.join(BASE_DIR, "combined_pneumothorax.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

df = pd.read_csv(CSV_PATH)
df['binary_label'] = df['Binary_Label']
df['path'] = df['Image_Index'].apply(lambda x: os.path.join(IMAGE_DIR, x))
""")

print(f"\n🚀 Ready to train with {len(combined_df):,} images!")