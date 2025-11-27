from typing import Dict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def save_checkpoint(state: Dict, filename: str):
    torch.save(state, filename)


def load_image(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.clamp(0.0, 1.0).detach().cpu()[0]
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def reconstruct_from_residual(input_tensor: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return (input_tensor - residual).clamp(0.0, 1.0)


def create_gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window_1d = (g / g.sum()).unsqueeze(0)
    window_2d = (window_1d.t() @ window_1d).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5):
    # expects input in [0,1]
    channels = img1.size(1)
    device = img1.device
    window = create_gaussian_window(window_size, sigma, channels, device).to(img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1
        vgg = models.vgg19(weights=weights).features
        self.capture_layers = {1, 6, 11, 20, 29}
        self.model = vgg.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.capture_layers:
                feats.append(x)
        return feats


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGG19FeatureExtractor()
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        feats_x = self.vgg(x.float())
        feats_y = self.vgg(y.float())
        loss = 0.0
        for fx, fy in zip(feats_x, feats_y):
            loss = loss + self.criterion(fx, fy)
        return loss
