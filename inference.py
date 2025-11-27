"""
Inference script for MSRF-NAFNet
Process single images or entire directories
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml

from model import create_msrf_nafnet_s, create_msrf_nafnet_m, create_msrf_nafnet_l
from utils import load_checkpoint


class Inferencer:
    """High-performance inference engine"""
    
    def __init__(self, model_path, config_path=None, device='cuda', use_amp=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and (device == 'cuda')
        
        # Load config if provided
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            model_type = config['model']['type']
        else:
            # Default to small model
            model_type = 'msrf_nafnet_s'
        
        # Create model
        if model_type == 'msrf_nafnet_s':
            self.model = create_msrf_nafnet_s()
        elif model_type == 'msrf_nafnet_m':
            self.model = create_msrf_nafnet_m()
        elif model_type == 'msrf_nafnet_l':
            self.model = create_msrf_nafnet_l()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'ema_state_dict' in checkpoint:
            # Use EMA weights if available
            state_dict = checkpoint['ema_state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
    
    def load_image(self, path):
        """Load image from path"""
        img = Image.open(path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return img_tensor, img
    
    def save_image(self, tensor, path):
        """Save tensor as image"""
        img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        img.save(path)
    
    def pad_to_multiple(self, img, multiple=32):
        """Pad image to be divisible by multiple"""
        _, _, h, w = img.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        
        return img, (h, w)
    
    def unpad(self, img, original_size):
        """Remove padding"""
        h, w = original_size
        return img[:, :, :h, :w]
    
    @torch.no_grad()
    def process_image(self, img_tensor):
        """Process single image"""
        # Pad to multiple of 32
        img_padded, original_size = self.pad_to_multiple(img_tensor)
        img_padded = img_padded.to(self.device)
        
        # Inference with mixed precision
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                output = self.model(img_padded)
        else:
            output = self.model(img_padded)
        
        # Remove padding
        output = self.unpad(output, original_size)
        
        # Clamp to valid range
        output = torch.clamp(output, 0, 1)
        
        return output
    
    def process_file(self, input_path, output_path):
        """Process single file"""
        img_tensor, original_img = self.load_image(input_path)
        output = self.process_image(img_tensor)
        self.save_image(output, output_path)
    
    def process_directory(self, input_dir, output_dir, recursive=False):
        """Process entire directory"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        
        if recursive:
            image_files = []
            for ext in extensions:
                image_files.extend(input_dir.rglob(f'*{ext}'))
        else:
            image_files = []
            for ext in extensions:
                image_files.extend(input_dir.glob(f'*{ext}'))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            # Determine output path
            if recursive:
                rel_path = img_path.relative_to(input_dir)
                out_path = output_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = output_dir / img_path.name
            
            # Process
            try:
                self.process_file(img_path, out_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    def process_with_tiles(self, img_tensor, tile_size=512, overlap=32):
        """
        Process large image with tiling for better quality and memory efficiency
        Useful for very large astronomical images
        """
        _, _, h, w = img_tensor.shape
        
        # If image is small enough, process normally
        if h <= tile_size and w <= tile_size:
            return self.process_image(img_tensor)
        
        # Create output tensor
        output = torch.zeros_like(img_tensor)
        weight = torch.zeros_like(img_tensor)
        
        # Calculate stride
        stride = tile_size - overlap
        
        # Process tiles
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                # Extract tile
                i_end = min(i + tile_size, h)
                j_end = min(j + tile_size, w)
                tile = img_tensor[:, :, i:i_end, j:j_end]
                
                # Pad if necessary
                tile_h, tile_w = tile.shape[2:]
                if tile_h < tile_size or tile_w < tile_size:
                    pad_h = tile_size - tile_h
                    pad_w = tile_size - tile_w
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                
                # Process tile
                tile_output = self.process_image(tile)
                
                # Remove padding
                tile_output = tile_output[:, :, :tile_h, :tile_w]
                
                # Add to output with weights
                output[:, :, i:i_end, j:j_end] += tile_output.cpu()
                weight[:, :, i:i_end, j:j_end] += 1
        
        # Normalize by weights
        output = output / weight.clamp(min=1)
        
        return output


def main():
    parser = argparse.ArgumentParser(description='MSRF-NAFNet Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output image or directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--recursive', action='store_true', help='Process directory recursively')
    parser.add_argument('--tile-size', type=int, default=0, help='Use tiling for large images (0=disable)')
    parser.add_argument('--tile-overlap', type=int, default=32, help='Overlap between tiles')
    
    args = parser.parse_args()
    
    # Create inferencer
    inferencer = Inferencer(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        use_amp=args.amp
    )
    
    # Process input
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Single file
        print(f"Processing single image: {input_path}")
        
        if args.tile_size > 0:
            img_tensor, _ = inferencer.load_image(input_path)
            output = inferencer.process_with_tiles(
                img_tensor,
                tile_size=args.tile_size,
                overlap=args.tile_overlap
            )
            inferencer.save_image(output, output_path)
        else:
            inferencer.process_file(input_path, output_path)
        
        print(f"Saved to: {output_path}")
    
    elif input_path.is_dir():
        # Directory
        print(f"Processing directory: {input_path}")
        inferencer.process_directory(
            input_dir=input_path,
            output_dir=output_path,
            recursive=args.recursive
        )
        print(f"Results saved to: {output_path}")
    
    else:
        print(f"Error: {input_path} not found")


if __name__ == '__main__':
    main()
