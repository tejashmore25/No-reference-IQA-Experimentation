import os
import sys
import torch
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms

# Assuming the official CONTRIQUE repo is cloned in the same directory
# sys.path.append(os.path.abspath('CONTRIQUE'))
from contrique.CONTRIQUE.modules.network import get_network
from contrique.CONTRIQUE.modules.CONTRIQUE_model import CONTRIQUE_model

class ContriqueEvaluator:
    """
    A modular wrapper for the CONTRIQUE IQA model, designed for comparative 
    research, feature extraction, and dynamic stress-testing.
    """
    def __init__(self, ckpt_path="CONTRIQUE/models/CONTRIQUE_checkpoint25.tar", 
                 regressor_path="CONTRIQUE/models/CLIVE.save", device=None):
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.ToTensor()
        
        # 1. Load the PyTorch Backbone (Encoder)
        self.model = self._load_backbone(ckpt_path)
        
        # Expose the encoder explicitly for Grad-CAM hooks
        # You can attach hooks to `self.encoder.layer4` for spatial feature maps
        self.encoder = self.model.encoder 
        
        # 2. Load the SVR Regressor
        self.regressor = self._load_regressor(regressor_path)

    def _load_backbone(self, ckpt_path):
        """Initializes the ResNet50 encoder and Contrastive projection head."""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint missing at {ckpt_path}. Please download it.")
            
        class DummyArgs:
            device = self.device
            
        encoder = get_network('resnet50', pretrained=False)
        # resnet50 is the encoder and 2048 is the feature dimension
        model = CONTRIQUE_model(DummyArgs(), encoder, 2048)
        
        state_dict = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def _load_regressor(self, regressor_path):
        """Loads the pre-trained Support Vector Regressor (SVR)."""
        if not os.path.exists(regressor_path):
            raise FileNotFoundError(f"Regressor missing at {regressor_path}.")
            
        with open(regressor_path, 'rb') as f:
            regressor = pickle.load(f)
        return regressor

    def _prepare_multiscale_tensors(self, image: Image.Image, requires_grad=False):
        """
        Prepares the original and downscaled tensors.
        Supports requires_grad=True if you attempt gradient-based visualizations.
        """
        w, h = image.size
        # The paper specifically uses Bicubic downscaling by a factor of 2
        image_2 = image.resize((w // 2, h // 2), Image.BICUBIC)
        
        img_t = self.transform(image).unsqueeze(0).to(self.device)
        img_2_t = self.transform(image_2).unsqueeze(0).to(self.device)
        
        if requires_grad:
            img_t.requires_grad_(True)
            img_2_t.requires_grad_(True)
            
        return img_t, img_2_t

    def extract_features(self, image: Image.Image, return_tensor=False, requires_grad=False):
        """
        Extracts the 4096-dimensional multiscale feature vector.
        Use this directly for PCA, t-SNE, or training custom regressors.
        """
        img_t, img_2_t = self._prepare_multiscale_tensors(image, requires_grad)
        
        # Only use torch.no_grad() if we aren't doing backward passes for Grad-CAM
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        
        with context:
            # Indices 4 and 5 correspond to the pooled features for scale 1 (2048) and scale 2 (2048)
            # rest are used in training and not needed for inference
            _, _, _, _, feat_1, feat_2, _, _ = self.model(img_t, img_2_t)
            
        # for backprop
        if return_tensor:
            return torch.cat((feat_1, feat_2), dim=1)
        
        # for visualization
        return np.hstack((
            feat_1.detach().cpu().numpy(), 
            feat_2.detach().cpu().numpy()
        ))

    def predict(self, image: Image.Image):
        """
        Predicts the perceptual quality score of a given PIL Image.
        Ideal for dynamic stress testing and calculating absolute errors in a loop.
        """
        features = self.extract_features(image, return_tensor=False)
        score = self.regressor.predict(features)
        return float(score[0])
        # return score