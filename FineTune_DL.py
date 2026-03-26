import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import time

# 1. Setup
DATA_DIR = r"D:\archive\Project_Dataset\Train_Data"
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 3 # Keep this low for local CPU testing

def main():
    print("Setting up the Deep Learning Pipeline...")
    
    # Check if GPU is available (Fallback to CPU for the ThinkBook)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}\n")

    # 2. Data Transformations (Adding slight augmentation to prevent overfitting)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Flips the X-ray randomly to create more variety
        transforms.RandomRotation(10),     # Slight tilts
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load and Split Data (80% Train, 20% Validation)
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Modify ResNet18 for Fine-Tuning
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # We do NOT freeze the layers here. We want the gradients to flow all the way back!
    # Replace the final layer to output our 6 specific classes instead of 1000
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    # 5. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    # A smaller learning rate (1e-4) is crucial so we don't destroy the pre-trained weights
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 6. The Training Loop
    print("Beginning Training...\n")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # Clear old gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate error
            
            _, preds = torch.max(outputs, 1)
            loss.backward() # Backpropagation (The magic happens here)
            optimizer.step() # Update weights

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- VALIDATION PHASE ---
        model.eval() # Turn off dropout/batchnorm updates for testing
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        v_loss = val_loss / val_size
        v_acc = val_corrects.double() / val_size
        print(f"Val Loss:   {v_loss:.4f} Acc: {v_acc:.4f}\n")

    print("Training Complete!")
    # Save the custom-trained model
    torch.save(model.state_dict(), r"D:\archive\Project_Dataset\custom_xray_resnet.pth")

if __name__ == "__main__":
    main()