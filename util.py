# Cell 2: Custom Feature-Based Grad-CAM for CONTRIQUE
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

class FeatureGradCAM:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        # Target the final convolutional layer of the ResNet50 encoder
        self.target_layer = self.evaluator.encoder[7][-1].conv3
        
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture forward activations and backward gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, we want the first element
        self.gradients = grad_output[0]

    def generate(self, image_path):
        image = Image.open(image_path).convert('RGB')
        
        # Use your wrapper's built-in multiscale prep, ensuring gradients are tracked
        img_t, img_2_t = self.evaluator._prepare_multiscale_tensors(image, requires_grad=True)
        
        self.evaluator.model.zero_grad()
        
        # Forward pass through the backbone
        _, _, _, _, feat_1, feat_2, _, _ = self.evaluator.model(img_t, img_2_t)
        
        # Concatenate features (mimicking what your wrapper does for the SVR)
        combined_features = torch.cat((feat_1, feat_2), dim=1)
        
        # Target: The L2 norm (magnitude) of the combined feature vector
        target = torch.norm(combined_features)
        
        # Backward pass to get gradients
        target.backward()
        
        # --- Grad-CAM Computation ---
        # 1. Global average pool the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # 2. Weight the activations by the pooled gradients
        activations = self.activations.detach()[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # 3. Average the channels and apply ReLU
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        # 4. Normalize the heatmap
        if np.max(heatmap) == 0:
            heatmap = np.zeros_like(heatmap)
        else:
            heatmap /= np.max(heatmap)
            
        return image, heatmap

def gradcam_output(image, heatmap):
    """Overlays the heatmap on the original image."""
    img_np = np.array(image)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # Convert heatmap to RGB coloring
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose
    superimposed_img = heatmap_colored * 0.4 + img_np * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img

def load_distortion_mapping(live_base_path, distortion_folder):
    """
    Parses the info.txt file inside a specific LIVE distortion folder.
    Returns a dictionary mapping the distorted image to its reference and strength.
    """
    info_path = os.path.join(live_base_path, distortion_folder, "info.txt")
    mapping = {}
    
    if not os.path.exists(info_path):
        print(f"Error: Could not find {info_path}")
        return mapping

    with open(info_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Usually format is: dist_img ref_img parameter
            if len(parts) >= 3:
                dist_img = parts[1]
                ref_img = parts[0]
                strength = float(parts[2])
                
                if ref_img in mapping.keys():
                    mapping[ref_img].append((dist_img, strength))
                else:
                    mapping[ref_img] = [(dist_img, strength)] 
                
    print(f"Loaded mapping for {len(mapping)} images in '{distortion_folder}'.")
    return mapping

def experiment1(img_name, dist, dist_map, dataset_path, contrique):
    cam_generator = FeatureGradCAM(contrique)

    sorted_dist_image_list = sorted(dist_map[img_name], key=lambda x: x[1])
    sorted_dist_image_list = [x for x in sorted_dist_image_list if x[1] != 0.0]
    fig, axes = plt.subplots(2, len(sorted_dist_image_list) + 1, figsize=(22, 7))
    axes = axes.flatten()

    ref_img_path = os.path.join(dataset_path, 'refimgs', img_name)
    ref_img, ref_img_heatmap = cam_generator.generate(ref_img_path)
    ref_gradCam = gradcam_output(ref_img, ref_img_heatmap)
    score = contrique.predict(ref_img)

    axes[0].imshow(ref_img)
    axes[0].set_title(f"Org (Score: {score:.2f})")
    axes[0].axis('off')
    axes[1].imshow(ref_gradCam)
    axes[1].set_title("Grad_Cam (Org)")
    axes[1].axis('off')

    for idx, (dist_img, strength) in enumerate(sorted_dist_image_list):
        img_path = os.path.join(dataset_path, dist, dist_img)
        img, img_heatmap = cam_generator.generate(img_path)
        gradCam = gradcam_output(img, img_heatmap)
        score = contrique.predict(img)

        axes[(idx * 2) + 2].imshow(img)
        axes[(idx * 2) + 2].set_title(f"(Score: {score:.2f}, Str: {strength})")
        axes[(idx * 2) + 2].axis('off')
        axes[(idx * 2) + 3].imshow(gradCam)
        axes[(idx * 2) + 3].set_title("Grad_Cam")
        axes[(idx * 2) + 3].axis('off')
    
    plt.suptitle(f"Distortion: {dist}", fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return 

def experiment3_gradCam(top_k_df, dataset_path, contrique, title, isDistortion=False):
    cam_generator = FeatureGradCAM(contrique)
    rows = 2

    fig, axes = plt.subplots(rows, len(top_k_df), figsize=(22, 7))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top_k_df.iterrows()):
        img_path = os.path.join(dataset_path, row['image_name'])
        img, img_heatmap = cam_generator.generate(img_path)
        gradCam = gradcam_output(img, img_heatmap)
        # score = contrique.predict(img)

        axes[(idx * 2)].imshow(img)
        axes[(idx * 2)].set_title(f"Error: {row['raw_error']:.2f}" if not isDistortion else f"Rad/Qual: {row['strength']}, Err: {row['raw_error']:.2f}")
        axes[(idx * 2)].axis('off')
        axes[(idx * 2) + 1].imshow(gradCam)
        axes[(idx * 2) + 1].set_title("Grad_Cam")
        axes[(idx * 2) + 1].axis('off')
    
    plt.suptitle(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return 

def compare_distortion_exp3(img_name, blur_df, jpeg_df):
    plt.figure(figsize=(12, 5))

    # Plot Blur Sensitivity
    plt.subplot(1, 2, 1)
    radii, new_scores, baseline_score = blur_df.loc[blur_df['image_name'] == img_name, ['strength', 'new_score', 'baseline_score']].values.T.tolist()
    plt.plot(radii, new_scores, marker='o', label='CONTRIQUE Score')
    plt.axhline(y=baseline_score[0], color='r', linestyle='--', label='Baseline')
    plt.xlabel('Blur Radius')
    plt.ylabel('Predicted Score')
    plt.title('Sensitivity to Gaussian Blur')
    plt.legend()

    # Plot JPEG Sensitivity
    plt.subplot(1, 2, 2)
    qualities, new_scores, baseline_score = jpeg_df.loc[jpeg_df['image_name'] == img_name, ['strength', 'new_score', 'baseline_score']].values.T.tolist()
    plt.plot(qualities, new_scores, marker='o', color='green', label='CONTRIQUE Score')
    plt.axhline(y=baseline_score[0], color='r', linestyle='--', label='Baseline')
    plt.gca().invert_xaxis() # Lower quality = more distortion
    plt.xlabel('JPEG Quality Level')
    plt.ylabel('Predicted Score')
    plt.title('Sensitivity to JPEG Compression')
    plt.legend()

    plt.suptitle(f"Distortion Sensitivity for {img_name}")
    plt.tight_layout()
    plt.show()

class KonIQDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df
        self.image_dir = image_dir
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_name']
        true_score = row['MOS']
        img_path = os.path.join(self.image_dir, img_name)

        # Load and resize exactly how the CONTRIQUE wrapper does it
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        image_2 = image.resize((w // 2, h // 2), Image.BICUBIC)
        
        img_t = self.transform(image)
        img_2_t = self.transform(image_2)
        
        return img_t, img_2_t, true_score, img_name