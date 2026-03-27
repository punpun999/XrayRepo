import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. Master Paths 
TEST_IMAGE_PATH = r"C:\Users\rayya\Pictures\Screenshots\Screenshot 2026-03-27 103309.png"
MODEL_WEIGHTS = r"D:\archive\Project_Dataset\master_densenet_weights.pth"
NUM_CLASSES = 6

# Class Map to translate math back to English
CLASS_NAMES = {
    0: 'Broken Hand', 1: 'Healthy Hand', 
    2: 'Broken Shoulder', 3: 'Healthy Shoulder', 
    4: 'Broken Wrist', 5: 'Healthy Wrist'
}

# 2. The Grad-CAM Engine
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook into the layer to grab the data during the forward/backward pass
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        
        # Calculate gradients for the predicted class
        target = output[0, pred_class]
        target.backward()

        # Weight the feature maps
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]

        # Generate the heatmap
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        
        # Prevent divide-by-zero errors
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap, pred_class

def main():
    print("Loading DenseNet121 Medical Expert...")
    device = torch.device("cpu") # CPU is perfect for single-image inference
    
    # 3. Rebuild the exact architecture
    model = models.densenet121()
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Load your hard-earned weights
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    # --- THE ULTIMATE INPLACE FIX (MONKEY PATCH) ---
    # PyTorch hardcoded 'inplace=True'. We hijack the forward pass to force it False.
    def custom_forward(x):
        features = model.features(x)
        out = torch.nn.functional.relu(features, inplace=False) # The magic fix
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = model.classifier(out)
        return out

    model.forward = custom_forward
    # -----------------------------------------------

    # Target the very last batch norm layer before the classifier
    target_layer = model.features.norm5
    cam = GradCAM(model, target_layer)

    # 4. Prepare the Image exactly how it was trained
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Analyzing Image: {TEST_IMAGE_PATH}")
    original_img = Image.open(TEST_IMAGE_PATH).convert('RGB')
    input_tensor = transform(original_img).unsqueeze(0)

    # 5. Generate the Heatmap
    heatmap, pred_class = cam.generate(input_tensor)
    prediction_text = CLASS_NAMES[pred_class]
    print(f"AI Diagnosis: {prediction_text}")

    # 6. Visualizing the Results
    # Resize heatmap to match the original image
    heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on the original image
    original_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    superimposed_img = heatmap * 0.4 + original_cv * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Display side-by-side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_img)
    ax[0].set_title("Original X-Ray")
    ax[0].axis('off')

    ax[1].imshow(superimposed_img)
    ax[1].set_title(f"AI Heatmap\nDiagnosis: {prediction_text}")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()