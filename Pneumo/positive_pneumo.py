import pandas as pd
import shutil
import os

# Read the CSV file
csv_path = './dataset_pneumothorax/pneumothorax.csv'
df = pd.read_csv(csv_path)

# Filter for positive labels (assuming 'label' column with 1 for positive)
positive_files = df[df['Binary_Label'] == 1]['Image_Index']

# Source and destination directories
source_dir = './dataset_pneumothorax/images'
dest_dir = './dataset_pneumothorax/images_positive'

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Copy each positive file
for file_name in positive_files:
    source_path = os.path.join(source_dir, file_name)
    dest_path = os.path.join(dest_dir, file_name)
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
        print(f"Copied {file_name}")
    else:
        print(f"File {file_name} not found in source directory")