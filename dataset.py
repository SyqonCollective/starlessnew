"""
Dataset loader optimized for RTX 5090
High-performance data loading with advanced augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
import random
import numpy as np
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class StarRemovalDataset(Dataset):
    """
    Star removal dataset with advanced augmentation
    Optimized for high-quality texture preservation
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        patch_size: int = 512,
        augment: bool = False,  # Dataset già con augmentations
        normalize: bool = True
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')
        self.normalize = normalize
        
        # Get image paths
        self.input_dir = self.root_dir / split / 'input'
        self.target_dir = self.root_dir / split / 'target'
        
        # Support multiple image formats
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        self.image_paths = []
        
        for ext in self.image_extensions:
            self.image_paths.extend(sorted(self.input_dir.glob(f'*{ext}')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.input_dir}")
        
        print(f"Found {len(self.image_paths)} images in {split} set")
        
        # Setup augmentation pipeline
        self.setup_augmentation()
    
    def setup_augmentation(self):
        """Setup albumentations pipeline for high-quality augmentation"""
        if self.augment:
            self.transform = A.Compose([
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.3
                ),
                
                # Color augmentations (subtle for astronomy)
                A.OneOf([
                    A.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.02,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=15,
                        val_shift_limit=10,
                        p=1.0
                    ),
                ], p=0.3),
                
                # Noise augmentation (realistic for astronomical images)
                A.OneOf([
                    A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
                    A.ISONoise(
                        color_shift=(0.01, 0.05),
                        intensity=(0.1, 0.3),
                        p=1.0
                    ),
                ], p=0.2),
                
                # Quality degradation
                A.OneOf([
                    A.Blur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.1),
                
            ], additional_targets={'target': 'image'})
        else:
            self.transform = None
    
    def load_image(self, path: Path) -> np.ndarray:
        """Load image and convert to numpy array"""
        img = Image.open(path).convert('RGB')
        return np.array(img)
    
    def random_crop(self, input_img: np.ndarray, target_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random crop both images to patch_size"""
        h, w = input_img.shape[:2]
        
        if h < self.patch_size or w < self.patch_size:
            # Pad if image is smaller than patch size
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            input_img = np.pad(input_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            target_img = np.pad(target_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w = input_img.shape[:2]
        
        # Random crop
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        
        input_patch = input_img[top:top+self.patch_size, left:left+self.patch_size]
        target_patch = target_img[top:top+self.patch_size, left:left+self.patch_size]
        
        return input_patch, target_patch
    
    def center_crop(self, input_img: np.ndarray, target_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Center crop for validation"""
        h, w = input_img.shape[:2]
        
        if h < self.patch_size or w < self.patch_size:
            # Pad if needed
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            input_img = np.pad(input_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            target_img = np.pad(target_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w = input_img.shape[:2]
        
        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        
        input_patch = input_img[top:top+self.patch_size, left:left+self.patch_size]
        target_patch = target_img[top:top+self.patch_size, left:left+self.patch_size]
        
        return input_patch, target_patch
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get paths
        input_path = self.image_paths[idx]
        target_path = self.target_dir / input_path.name
        
        # Load images (già tile 512x512 con augmentations)
        input_img = self.load_image(input_path)
        target_img = self.load_image(target_path)
        
        # Nessun crop necessario - immagini già alla dimensione corretta
        # Nessuna augmentation - già fatta durante preparazione dataset
        
        # Convert to tensor and normalize to [0, 1]
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1).float() / 255.0
        
        return input_tensor, target_tensor


class PrefetchLoader:
    """
    Prefetch loader for faster data loading on GPU
    Optimized for RTX 5090
    """
    def __init__(self, loader, device='cuda'):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        
    def __iter__(self):
        first = True
        
        for next_input, next_target in self.loader:
            with torch.cuda.stream(self.stream):
                next_input = next_input.to(self.device, non_blocking=True)
                next_target = next_target.to(self.device, non_blocking=True)
            
            if not first:
                yield input_batch, target_batch
            else:
                first = False
            
            torch.cuda.current_stream().wait_stream(self.stream)
            input_batch = next_input
            target_batch = next_target
        
        yield input_batch, target_batch
    
    def __len__(self):
        return len(self.loader)


def create_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 8,
    patch_size: int = 512,
    pin_memory: bool = True,
    prefetch: bool = True,
    device: str = 'cuda'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        root_dir: Root directory containing train/val folders
        batch_size: Batch size (optimize for RTX 5090 VRAM)
        num_workers: Number of worker processes
        patch_size: Size of image patches
        pin_memory: Use pinned memory for faster GPU transfer
        prefetch: Use prefetch loader for better performance
        device: Device for prefetching
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets (no augmentation - dataset già preparato)
    train_dataset = StarRemovalDataset(
        root_dir=root_dir,
        split='train',
        patch_size=patch_size,
        augment=False  # Dataset già con augmentations
    )
    
    val_dataset = StarRemovalDataset(
        root_dir=root_dir,
        split='val',
        patch_size=patch_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Wrap with prefetch loader if requested
    if prefetch and device == 'cuda':
        train_loader = PrefetchLoader(train_loader, device=device)
        val_loader = PrefetchLoader(val_loader, device=device)
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset
    dataset = StarRemovalDataset(
        root_dir='/Users/michaelruggeri/Desktop/starless',
        split='train',
        patch_size=256
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading
    input_img, target_img = dataset[0]
    print(f"Input shape: {input_img.shape}")
    print(f"Target shape: {target_img.shape}")
    print(f"Input range: [{input_img.min():.3f}, {input_img.max():.3f}]")
    print(f"Target range: [{target_img.min():.3f}, {target_img.max():.3f}]")
