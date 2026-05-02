import os
import shutil
from pathlib import Path

def move_all_images_to_single_folder():
    """
    Move all PNG images from dataset_unbalanced/images_XXX/images/ folders
    to dataset_unbalanced/images/
    """
    
    # Define paths
    base_dir = Path("dataset_unbalanced")
    destination_dir = base_dir / "images"
    
    # Create destination directory if it doesn't exist
    destination_dir.mkdir(exist_ok=True)
    
    print(f"Moving images to: {destination_dir}")
    print("=" * 50)
    
    total_moved = 0
    
    # Look for all images_XXX folders
    for images_folder in base_dir.glob("images_*"):
        if images_folder.is_dir():
            # Path to the nested images folder
            nested_images_folder = images_folder / "images"
            
            if nested_images_folder.exists() and nested_images_folder.is_dir():
                print(f"Processing: {images_folder.name}/images/")
                
                # Count PNG files in this folder
                png_files = list(nested_images_folder.glob("*.png"))
                folder_count = len(png_files)
                
                print(f"  Found {folder_count} PNG files")
                
                # Move each PNG file
                for png_file in png_files:
                    destination_path = destination_dir / png_file.name
                    
                    # Handle duplicate filenames
                    counter = 1
                    original_name = png_file.stem
                    while destination_path.exists():
                        new_name = f"{original_name}_{counter}.png"
                        destination_path = destination_dir / new_name
                        counter += 1
                    
                    # Move the file
                    try:
                        shutil.move(str(png_file), str(destination_path))
                        total_moved += 1
                    except Exception as e:
                        print(f"  Error moving {png_file.name}: {e}")
                
                print(f"  Moved {folder_count} files from {images_folder.name}")
                
                # Optional: Remove empty nested folders
                try:
                    if not list(nested_images_folder.iterdir()):  # If folder is empty
                        nested_images_folder.rmdir()
                        print(f"  Removed empty folder: {nested_images_folder}")
                except:
                    pass
            else:
                print(f"No images folder found in: {images_folder.name}")
    
    print("=" * 50)
    print(f"Total images moved: {total_moved}")
    print(f"All images are now in: {destination_dir}")
    
    # Verify the final count
    final_count = len(list(destination_dir.glob("*.png")))
    print(f"Final count in destination: {final_count} PNG files")

def verify_folder_structure():
    """
    Verify the current folder structure before moving
    """
    base_dir = Path("dataset_unbalanced")
    
    if not base_dir.exists():
        print(f"Error: {base_dir} folder not found!")
        return False
    
    print("Current folder structure:")
    print("=" * 30)
    
    total_images = 0
    
    for images_folder in sorted(base_dir.glob("images_*")):
        if images_folder.is_dir():
            nested_images_folder = images_folder / "images"
            if nested_images_folder.exists():
                png_count = len(list(nested_images_folder.glob("*.png")))
                total_images += png_count
                print(f"{images_folder.name}/images/ - {png_count} PNG files")
            else:
                print(f"{images_folder.name}/ - No images subfolder found")
    
    print("=" * 30)
    print(f"Total PNG files found: {total_images}")
    
    return total_images > 0

if __name__ == "__main__":
    print("Chest X-ray Image Consolidation Script")
    print("=" * 40)
    
    # First verify the structure
    if verify_folder_structure():
        print("\nProceeding with image consolidation...")
        response = input("\nDo you want to continue? (y/n): ").lower().strip()
        
        if response == 'y' or response == 'yes':
            move_all_images_to_single_folder()
            print("\n✅ Image consolidation completed!")
        else:
            print("Operation cancelled.")
    else:
        print("❌ No valid folder structure found. Please check the dataset_unbalanced folder.")