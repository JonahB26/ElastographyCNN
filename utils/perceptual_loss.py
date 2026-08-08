"""
Perceptual Loss using VGG16 features.

Usage:
    from utils.perceptual_loss import PerceptualLoss
    
    perceptual = PerceptualLoss(weight=0.1).to(device)
    
    # In your loss computation:
    loss = base_loss(pred, target) + perceptual(pred, target)
    
Note: VGG expects 3-channel input. This module repeats the single-channel
elastography image to 3 channels automatically.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 conv features.
    
    Extracts features at multiple layers and computes L1 distance
    between prediction and target features. Deeper layers capture
    higher-level structure; shallower layers capture texture/edges.
    
    Args:
        weight: overall weight for the perceptual loss term
        layers: which VGG layers to extract features from
                (default: after conv2_2, conv3_3, conv4_3)
        normalize: whether to normalize input to VGG's expected range
    """
    
    def __init__(self, weight=0.1, layers=None, normalize=True):
        super().__init__()
        self.weight = weight
        self.normalize = normalize
        
        # Load pretrained VGG16 and freeze it
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        if layers is None:
            layers = [9, 16, 23]
        self.layers = layers
        
        max_layer = max(layers) + 1
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer])
        
        # Freeze all VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # VGG normalization constants (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def extract_features(self, x):
        """Extract features at the specified layers."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        if self.normalize:
            x = (x - self.mean) / self.std
        
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        
        return features
    
    def forward(self, pred, target):
        """Compute perceptual loss between prediction and target."""
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = torch.tensor(0.0, device=pred.device)
        for pf, tf in zip(pred_features, target_features):
            loss = loss + nn.functional.l1_loss(pf, tf)
        
        return self.weight * loss