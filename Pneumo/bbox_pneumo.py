import pandas as pd
import os

# Paths
input_csv = os.path.join("dataset_pneumothorax", "BBox_List_2017.csv")
output_csv = os.path.join("dataset_pneumothorax", "BBox_List_2017_pneumothorax_any.csv")

# Read the CSV
df = pd.read_csv(input_csv)

# Filter: include any row where 'Finding Label' contains 'pneumothorax' (case-insensitive, even if multi-label)
df_pneumo = df[df["Finding Label"].str.lower().str.contains("pneumothorax")]

# Save to new CSV
df_pneumo.to_csv(output_csv, index=False)

print(f"Saved {len(df_pneumo)} rows with pneumothorax (including multi-label) to {output_csv}")