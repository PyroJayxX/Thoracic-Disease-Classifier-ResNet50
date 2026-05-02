import pandas as pd
import os
import shutil
from pathlib import Path

# This is a script for creating a new dataset with only 5 disease classes plus No Finding from balanced_dataset

# Define paths
source_images_dir = './dataset_balanced/new_images/new_images/'
source_csv = './dataset_balanced/new_labels.csv'
output_dir = './dataset_5nodes/'
output_images_dir = os.path.join(output_dir, 'images')
output_csv = os.path.join(output_dir, 'labels.csv')

# Classes to keep
classes_to_keep = [
    'Consolidation',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'No Finding'
]

# Create output directories
os.makedirs(output_images_dir, exist_ok=True)

# Read the CSV
print("Reading CSV file...")
df = pd.read_csv(source_csv)

# Filter rows that have at least one of the desired classes
print("Filtering dataset...")
mask = df[classes_to_keep].sum(axis=1) > 0
filtered_df = df[mask].copy()

# Keep only the Path column and the desired class columns
columns_to_keep = ['Path'] + classes_to_keep
filtered_df = filtered_df[columns_to_keep]

# Copy images and track progress
print(f"\nCopying {len(filtered_df)} images...")
copied_count = 0
missing_count = 0

for idx, row in filtered_df.iterrows():
    image_name = row['Path']
    source_path = os.path.join(source_images_dir, image_name)
    dest_path = os.path.join(output_images_dir, image_name)
    
    if os.path.exists(source_path):
        shutil.copy2(source_path, dest_path)
        copied_count += 1
        if copied_count % 1000 == 0:
            print(f"  Copied {copied_count}/{len(filtered_df)} images...")
    else:
        print(f"  Warning: Image not found - {image_name}")
        missing_count += 1

# Save the filtered CSV
print("\nSaving filtered CSV...")
filtered_df.to_csv(output_csv, index=False)

# Print statistics
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Total images copied: {copied_count}")
print(f"Missing images: {missing_count}")
print(f"\nClass distribution:")
for class_name in classes_to_keep:
    count = filtered_df[class_name].sum()
    percentage = (count / len(filtered_df)) * 100
    print(f"  {class_name:20s}: {count:5d} samples ({percentage:5.2f}%)")

print(f"\nOutput location:")
print(f"  Images: {output_images_dir}")
print(f"  CSV: {output_csv}")
print("="*50)