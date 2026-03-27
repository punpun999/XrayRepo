import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import copy
import time

# 1. Setup Your Master Paths
TRAIN_DIR = r"D:\archive\MURA-v1.1\train" 
VALID_DIR = r"D:\archive\MURA-v1.1\valid" 

NUM_CLASSES = 6
BATCH_SIZE = 32
MAX_EPOCHS = 20 
PATIENCE = 4    

# 2. Build the Custom PyTorch Data Loader
class MURADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.class_map = {
            'XR_HAND_positive': 0,      
            'XR_HAND_negative': 1,      
            'XR_SHOULDER_positive': 2,  
            'XR_SHOULDER_negative': 3,  
            'XR_WRIST_positive': 4,     
            'XR_WRIST_negative': 5      
        }

        print(f"Scanning {root_dir} for valid images...")
        
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.png') and not file.startswith('._'):
                    path = os.path.join(subdir, file)
                    if any(part in path for part in ['XR_HAND', 'XR_SHOULDER', 'XR_WRIST']):
                        if 'XR_HAND' in path: part = 'XR_HAND'
                        elif 'XR_SHOULDER' in path: part = 'XR_SHOULDER'
                        else: part = 'XR_WRIST'

                        condition = 'positive' if 'positive' in path else 'negative'
                        dict_key = f"{part}_{condition}"
                        
                        self.image_paths.append(path)
                        self.labels.append(self.class_map[dict_key])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    print("Initializing Intel Arc Accelerated Pipeline...")
    # Checking for Intel GPU
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Compute Device: {device}\n")

    # 3. Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = MURADataset(TRAIN_DIR, transform=train_transform)
    val_dataset = MURADataset(VALID_DIR, transform=val_transform)

    # num_workers=4 uses your CPU to feed the GPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"\nSuccessfully loaded {len(train_dataset)} Training Images.")
    print(f"Successfully loaded {len(val_dataset)} Validation Images.\n")

    # 4. Architecture: DenseNet121
    print("Building DenseNet121 Architecture...")
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    # 5. Optimizer, Loss, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # 6. The Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    print("Starting Deep Learning Loop...")
    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")
        print("-" * 20)
        
        # --- TRAINING ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # The Correct Modern Autocast Syntax
            if device.type == 'xpu':
                with torch.amp.autocast(device_type='xpu'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                if device.type == 'xpu':
                    with torch.amp.autocast(device_type='xpu'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        v_loss = val_loss / len(val_dataset)
        v_acc = val_corrects.double() / len(val_dataset)
        print(f"Val Loss:   {v_loss:.4f} | Acc: {v_acc:.4f}")

        scheduler.step(v_loss)

        if v_acc > best_acc:
            best_acc = v_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("  [!] New best model saved.")
        else:
            epochs_no_improve += 1
            print(f"  [-] No accuracy improvement. Patience: {epochs_no_improve}/{PATIENCE}")

        if epochs_no_improve >= PATIENCE:
            print("\n*** Early stopping triggered! ***")
            break

    # 7. Wrap Up
    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Absolute Best Validation Accuracy: {best_acc:.4f}")
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), r"D:\archive\Project_Dataset\master_densenet_weights.pth")
    print("Master weights successfully saved to disk.")

if __name__ == "__main__":
    main()