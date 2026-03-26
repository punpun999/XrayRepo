import os
import random
import shutil

# Set your paths based on your D: drive structure
SOURCE_DIR = r"D:\archive\MURA-v1.1\train"
DEST_DIR = r"D:\archive\Project_Dataset\Train_Data"

# The 3 body parts we care about for this project (Swapped Finger for Shoulder)
TARGET_PARTS = ["XR_HAND", "XR_WRIST", "XR_SHOULDER"]

# How many images per category we want to extract
SAMPLES_PER_CLASS = 1000

def organize_and_sample_data():
    os.makedirs(DEST_DIR, exist_ok=True)
    
    for part in TARGET_PARTS:
        part_path = os.path.join(SOURCE_DIR, part)
        if not os.path.exists(part_path):
            print(f"Warning: Could not find {part_path}")
            continue

        positive_images = []
        negative_images = []

        # Walk through the nested patient and study folders
        for root, dirs, files in os.walk(part_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    # Check if the folder path says it's broken (positive) or healthy (negative)
                    if 'positive' in root:
                        positive_images.append(full_path)
                    elif 'negative' in root:
                        negative_images.append(full_path)

        # Randomly sample the images
        sampled_positives = random.sample(positive_images, min(SAMPLES_PER_CLASS, len(positive_images)))
        sampled_negatives = random.sample(negative_images, min(SAMPLES_PER_CLASS, len(negative_images)))

        # Create our clean, flat categories
        clean_part_name = part.replace("XR_", "").capitalize() 
        
        folders_to_create = {
            f"{clean_part_name}_Broken": sampled_positives,
            f"{clean_part_name}_Healthy": sampled_negatives
        }

        # Copy the files over
        for class_name, file_paths in folders_to_create.items():
            class_dir = os.path.join(DEST_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            print(f"Copying {len(file_paths)} images to {class_name}...")
            for i, file_path in enumerate(file_paths):
                new_filename = f"{class_name}_{i}.png" 
                shutil.copy(file_path, os.path.join(class_dir, new_filename))

    print("\nData organization complete! Your training subset is ready.")

if __name__ == "__main__":
    organize_and_sample_data()