"""
Advanced loss functions for genuine texture reconstruction
Combines multiple losses to avoid blob artifacts and preserve detail
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features
    Critical for texture quality and avoiding blob artifacts
    """
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'], weights=[1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.layers = layers
        self.weights = weights
        
        # Create feature extractors
        self.feature_extractors = nn.ModuleDict()
        
        layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_4': 17,
            'relu4_4': 26,
            'relu5_4': 35
        }
        
        for name in layers:
            if name in layer_map:
                extractor = nn.Sequential()
                for i in range(layer_map[name] + 1):
                    extractor.add_module(str(i), vgg[i])
                self.feature_extractors[name] = extractor
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Normalization for VGG
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize input for VGG"""
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        # Normalize inputs
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        # Calculate loss for each layer
        loss = 0.0
        for layer_name, weight in zip(self.layers, self.weights):
            pred_feat = self.feature_extractors[layer_name](pred)
            target_feat = self.feature_extractors[layer_name](target)
            loss += weight * F.l1_loss(pred_feat, target_feat)
        
        return loss


class TextureLoss(nn.Module):
    """
    Texture loss based on Gram matrices
    Ensures generated texture has similar statistics to target
    """
    def __init__(self):
        super().__init__()
        
    def gram_matrix(self, x):
        """Calculate Gram matrix"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred, target):
        gram_pred = self.gram_matrix(pred)
        gram_target = self.gram_matrix(target)
        return F.mse_loss(gram_pred, gram_target)


class EdgeLoss(nn.Module):
    """
    Edge-aware loss for detail preservation
    Helps maintain sharp edges and fine details
    """
    def __init__(self):
        super().__init__()
        
        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).float().view(1, 1, 3, 3))
    
    def get_edges(self, x):
        """Extract edges using Sobel filters"""
        # Convert to grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Apply Sobel filters
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Magnitude
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edges
    
    def forward(self, pred, target):
        edges_pred = self.get_edges(pred)
        edges_target = self.get_edges(target)
        return F.l1_loss(edges_pred, edges_target)


class FrequencyLoss(nn.Module):
    """
    Frequency domain loss for texture consistency
    Ensures frequency content matches between prediction and target
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # Loss on magnitude
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        return F.l1_loss(pred_mag, target_mag)


class CombinedLoss(nn.Module):
    """
    Combined loss function for high-quality star removal
    
    Weights optimized to:
    - Preserve genuine texture (perceptual + texture loss)
    - Avoid blob artifacts (edge + frequency loss)
    - Maintain color accuracy (L1 loss)
    """
    def __init__(
        self,
        l1_weight=1.0,
        perceptual_weight=1.0,
        texture_weight=0.5,
        edge_weight=0.3,
        frequency_weight=0.2
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.texture_weight = texture_weight
        self.edge_weight = edge_weight
        self.frequency_weight = frequency_weight
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.texture_loss = TextureLoss()
        self.edge_loss = EdgeLoss()
        self.frequency_loss = FrequencyLoss()
    
    def forward(self, pred, target, return_components=False):
        # Calculate individual losses
        l1 = self.l1_loss(pred, target) * self.l1_weight
        perceptual = self.perceptual_loss(pred, target) * self.perceptual_weight
        texture = self.texture_loss(pred, target) * self.texture_weight
        edge = self.edge_loss(pred, target) * self.edge_weight
        frequency = self.frequency_loss(pred, target) * self.frequency_weight
        
        # Total loss
        total_loss = l1 + perceptual + texture + edge + frequency
        
        if return_components:
            return total_loss, {
                'total': total_loss.item(),
                'l1': l1.item(),
                'perceptual': perceptual.item(),
                'texture': texture.item(),
                'edge': edge.item(),
                'frequency': frequency.item()
            }
        
        return total_loss


if __name__ == '__main__':
    # Test losses
    pred = torch.randn(2, 3, 256, 256).cuda()
    target = torch.randn(2, 3, 256, 256).cuda()
    
    loss_fn = CombinedLoss().cuda()
    loss, components = loss_fn(pred, target, return_components=True)
    
    print("Loss components:")
    for name, value in components.items():
        print(f"  {name}: {value:.6f}")
