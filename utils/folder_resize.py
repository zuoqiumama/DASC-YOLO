import os
from PIL import Image
import glob

def resize_images(src_folder, dest_folder, size=(500, 500)):
    """
    Resize all images in src_folder and save them to dest_folder
    
    Args:
        src_folder: Source folder containing images
        dest_folder: Destination folder for resized images
        size: Tuple (width, height) for the target size
    """
    # Create destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(src_folder, ext)))
        image_files.extend(glob.glob(os.path.join(src_folder, ext.upper())))
    
    # Process each image
    for img_path in image_files:
        try:
            # Open image
            img = Image.open(img_path)
            
            # Resize image
            img_resized = img.resize(size, Image.LANCZOS)
            
            # Save resized image
            filename = os.path.basename(img_path)
            save_path = os.path.join(dest_folder, filename)
            img_resized.save(save_path)
            
            print(f"Resized: {filename}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Example usage
if __name__ == "__main__":
    source_folder = "/home/cigit/project/defect/all"  # Change this to your source folder
    target_folder = "/home/cigit/project/defect/all/1024"   # Change this to your target folder
    target_size = (1024, 1024)           # Change this to your desired size
    
    resize_images(source_folder, target_folder, target_size)