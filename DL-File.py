import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 1. Setup your paths
DATA_DIR = r"D:\archive\Project_Dataset\Train_Data"
OUTPUT_FILE = r"D:\archive\Project_Dataset\extracted_features.npz"

def setup_extractor():
    print("Loading pre-trained ResNet18...")
    # Load ResNet18 with standard pre-trained weights
    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)
    
    # THE TRICK: Replace the final classification layer with an "Identity" layer.
    # Now, instead of a prediction, the model just outputs the raw visual features.
    resnet.fc = nn.Identity()
    resnet.eval() # Lock the model (turns off training mode)
    return resnet

def main():
    resnet = setup_extractor()

    # 2. Define how to process the X-rays
    # ResNet expects images to be exactly 224x224 and normalized in a specific way
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Load the organized dataset
    # ImageFolder automatically figures out the 6 classes based on your folder names!
    dataset = ImageFolder(root=DATA_DIR, transform=preprocess)
    
    # Batch size 32 is a safe, efficient size for local CPU processing
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    print(f"\nFound {len(dataset)} images belonging to {len(dataset.classes)} classes.")
    print(f"Classes found: {dataset.classes}\n")
    
    all_features = []
    all_labels = []
    
    # 4. Run the Extraction
    # torch.no_grad() tells PyTorch we aren't training, which saves massive amounts of memory
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            # Pass the batch of images through our modified ResNet
            features = resnet(images)
            
            # Store the mathematical results and the labels
            all_features.append(features.numpy())
            all_labels.append(labels.numpy())
            
    # Combine everything into clean, final arrays
    final_features = np.concatenate(all_features, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    
    # 5. Save the data to disk
    # We save this so you only ever have to run this heavy extraction process once!
    np.savez(OUTPUT_FILE, features=final_features, labels=final_labels)
    
    print(f"\nExtraction complete!")
    print(f"Feature matrix saved to: {OUTPUT_FILE}")
    print(f"Final Data Shape: {final_features.shape[0]} images, {final_features.shape[1]} features each.")

if __name__ == "__main__":
    main()