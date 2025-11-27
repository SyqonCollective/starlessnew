"""
MSRF-NAFNet GUI - Desktop Application
Optimized for Apple Silicon (MPS)
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread
import os

# Import model architecture (copy from main project)
sys.path.insert(0, str(Path(__file__).parent))
from model import create_msrf_nafnet_s, create_msrf_nafnet_m, create_msrf_nafnet_l


class StarRemovalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MSRF-NAFNet - Star Removal")
        self.root.geometry("800x600")
        
        # Variables
        self.model = None
        self.device = self.get_device()
        self.checkpoint_path = tk.StringVar()
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.tile_size = tk.IntVar(value=512)
        self.overlap = tk.IntVar(value=100)
        self.model_type = tk.StringVar(value="msrf_nafnet_s")
        
        self.setup_ui()
        
    def get_device(self):
        """Detect best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def setup_ui(self):
        """Setup user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="MSRF-NAFNet Star Removal", 
                         font=("Helvetica", 18, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Device info
        device_label = ttk.Label(main_frame, 
                                text=f"Device: {self.device}", 
                                font=("Helvetica", 10))
        device_label.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Model type selection
        ttk.Label(main_frame, text="Model Type:").grid(row=2, column=0, sticky=tk.W, pady=5)
        model_combo = ttk.Combobox(main_frame, textvariable=self.model_type,
                                   values=["msrf_nafnet_s", "msrf_nafnet_m", "msrf_nafnet_l"],
                                   state="readonly", width=30)
        model_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Checkpoint selection
        ttk.Label(main_frame, text="Checkpoint:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.checkpoint_path, width=40).grid(
            row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_checkpoint).grid(
            row=3, column=2, padx=5, pady=5)
        
        # Load model button
        ttk.Button(main_frame, text="Load Model", command=self.load_model,
                  style="Accent.TButton").grid(row=4, column=1, pady=10)
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(
            row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Input image selection
        ttk.Label(main_frame, text="Input Image:").grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_path, width=40).grid(
            row=6, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input).grid(
            row=6, column=2, padx=5, pady=5)
        
        # Output path selection
        ttk.Label(main_frame, text="Output Path:").grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_path, width=40).grid(
            row=7, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(
            row=7, column=2, padx=5, pady=5)
        
        # Tile size
        ttk.Label(main_frame, text="Tile Size:").grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(main_frame, from_=256, to=2048, increment=128,
                   textvariable=self.tile_size, width=15).grid(
            row=8, column=1, sticky=tk.W, pady=5)
        
        # Overlap
        ttk.Label(main_frame, text="Overlap:").grid(row=9, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(main_frame, from_=32, to=256, increment=32,
                   textvariable=self.overlap, width=15).grid(
            row=9, column=1, sticky=tk.W, pady=5)
        
        # Process button
        self.process_btn = ttk.Button(main_frame, text="Remove Stars", 
                                     command=self.process_image,
                                     style="Accent.TButton", state="disabled")
        self.process_btn.grid(row=10, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=500)
        self.progress.grid(row=11, column=0, columnspan=3, pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", 
                                     font=("Helvetica", 10))
        self.status_label.grid(row=12, column=0, columnspan=3, pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def browse_checkpoint(self):
        """Browse for checkpoint file"""
        filename = filedialog.askopenfilename(
            title="Select Checkpoint",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        if filename:
            self.checkpoint_path.set(filename)
    
    def browse_input(self):
        """Browse for input image"""
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.input_path.set(filename)
            # Auto-suggest output path
            input_file = Path(filename)
            output_file = input_file.parent / f"{input_file.stem}_starless{input_file.suffix}"
            self.output_path.set(str(output_file))
    
    def browse_output(self):
        """Browse for output path"""
        filename = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("TIFF", "*.tif"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.output_path.set(filename)
    
    def load_model(self):
        """Load model from checkpoint"""
        checkpoint_file = self.checkpoint_path.get()
        if not checkpoint_file or not Path(checkpoint_file).exists():
            messagebox.showerror("Error", "Please select a valid checkpoint file")
            return
        
        try:
            self.status_label.config(text="Loading model...")
            self.progress.start()
            
            # Create model
            model_type = self.model_type.get()
            if model_type == "msrf_nafnet_s":
                self.model = create_msrf_nafnet_s()
            elif model_type == "msrf_nafnet_m":
                self.model = create_msrf_nafnet_m()
            elif model_type == "msrf_nafnet_l":
                self.model = create_msrf_nafnet_l()
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'ema_state_dict' in checkpoint:
                # Load EMA weights if available (better quality)
                state_dict = {}
                for key, value in checkpoint['ema_state_dict'].items():
                    state_dict[key] = value
                self.model.load_state_dict(state_dict)
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.progress.stop()
            self.status_label.config(text=f"Model loaded successfully on {self.device}")
            self.process_btn.config(state="normal")
            
            messagebox.showinfo("Success", "Model loaded successfully!")
            
        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="Error loading model")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def process_image(self):
        """Process image in separate thread"""
        input_file = self.input_path.get()
        output_file = self.output_path.get()
        
        if not input_file or not Path(input_file).exists():
            messagebox.showerror("Error", "Please select a valid input image")
            return
        
        if not output_file:
            messagebox.showerror("Error", "Please specify output path")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        # Run in thread to avoid freezing UI
        thread = Thread(target=self._process_image_thread, 
                       args=(input_file, output_file))
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self, input_file, output_file):
        """Process image with tiling"""
        try:
            print(f"\n{'='*60}")
            print(f"Starting image processing")
            print(f"Input: {input_file}")
            print(f"Output: {output_file}")
            print(f"{'='*60}")
            
            self.status_label.config(text="Processing image...")
            self.progress.start()
            self.process_btn.config(state="disabled")
            
            # Load image
            print("\nLoading image...")
            img = Image.open(input_file).convert('RGB')
            print(f"PIL Image mode: {img.mode}, size: {img.size}")
            img_np = np.array(img).astype(np.float32) / 255.0
            print(f"Numpy array shape: {img_np.shape}, dtype: {img_np.dtype}")
            print(f"Numpy array range: [{img_np.min():.4f}, {img_np.max():.4f}]")
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            print(f"Tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
            print(f"Tensor range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
            
            # Process with tiling
            tile_size = self.tile_size.get()
            overlap = self.overlap.get()
            
            output = self.process_with_tiling(img_tensor, tile_size, overlap)
            
            print(f"\n=== Saving Output ===")
            print(f"Output tensor shape: {output.shape}")
            print(f"Output tensor range: [{output.min():.4f}, {output.max():.4f}]")
            
            # Save result
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            print(f"Output numpy shape: {output_np.shape}, dtype: {output_np.dtype}")
            print(f"Output numpy range: [{output_np.min():.4f}, {output_np.max():.4f}]")
            
            output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
            print(f"Output numpy after scaling shape: {output_np.shape}, dtype: {output_np.dtype}")
            print(f"Output numpy after scaling range: [{output_np.min()}, {output_np.max()}]")
            
            output_img = Image.fromarray(output_np)
            print(f"PIL output mode: {output_img.mode}, size: {output_img.size}")
            output_img.save(output_file)
            print(f"Image saved to: {output_file}")
            print(f"{'='*60}\n")
            
            self.progress.stop()
            self.status_label.config(text="Processing complete!")
            self.process_btn.config(state="normal")
            
            messagebox.showinfo("Success", f"Image saved to:\n{output_file}")
            
        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="Error processing image")
            self.process_btn.config(state="normal")
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
    
    def process_with_tiling(self, img_tensor, tile_size=512, overlap=100):
        """
        Process large image with tiling and smooth blending
        Overlap ensures seamless transitions
        """
        _, _, h, w = img_tensor.shape
        print(f"\n=== Processing Image ===")
        print(f"Image size: {h}x{w}")
        print(f"Tile size: {tile_size}, Overlap: {overlap}")
        print(f"Input tensor shape: {img_tensor.shape}")
        print(f"Input tensor range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
        
        # If image is small, process directly
        if h <= tile_size and w <= tile_size:
            print("Image is small, processing directly without tiling")
            img_padded = self.pad_to_multiple(img_tensor, 32)
            print(f"Padded shape: {img_padded.shape}")
            img_padded = img_padded.to(self.device)
            
            with torch.no_grad():
                output = self.model(img_padded)
            
            print(f"Output shape: {output.shape}")
            print(f"Output range before unpad: [{output.min():.4f}, {output.max():.4f}]")
            output = self.unpad(output, (h, w))
            print(f"Output range after unpad: [{output.min():.4f}, {output.max():.4f}]")
            return torch.clamp(output, 0, 1)
        
        # Create output tensor and weight map
        output = torch.zeros_like(img_tensor)
        weight = torch.zeros_like(img_tensor)
        
        print("\n=== Using Tiled Processing ===")
        
        # Calculate stride
        stride = tile_size - overlap
        
        # Calculate total number of tiles
        n_tiles_h = len(range(0, h, stride))
        n_tiles_w = len(range(0, w, stride))
        total_tiles = n_tiles_h * n_tiles_w
        current_tile = 0
        
        print(f"Grid: {n_tiles_h}x{n_tiles_w} = {total_tiles} tiles")
        print(f"Stride: {stride}")
        
        # Create weight matrix for smooth blending
        weight_tile = self.create_weight_matrix(tile_size, overlap)
        weight_tile = weight_tile.to(self.device)
        print(f"Weight tile shape: {weight_tile.shape}")
        print(f"Weight tile range: [{weight_tile.min():.4f}, {weight_tile.max():.4f}]")
        
        # Process tiles
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                current_tile += 1
                print(f"\n--- Tile {current_tile}/{total_tiles} ---")
                self.status_label.config(text=f"Processing tile {current_tile}/{total_tiles}...")
                self.root.update_idletasks()
                # Extract tile with bounds checking
                i_end = min(i + tile_size, h)
                j_end = min(j + tile_size, w)
                
                # Adjust start if we're at the edge
                i_start = max(0, i_end - tile_size)
                j_start = max(0, j_end - tile_size)
                
                print(f"Position: row [{i_start}:{i_end}], col [{j_start}:{j_end}]")
                
                tile = img_tensor[:, :, i_start:i_end, j_start:j_end]
                print(f"Tile shape: {tile.shape}")
                print(f"Tile input range: [{tile.min():.4f}, {tile.max():.4f}]")
                
                # Store original tile for residual subtraction
                tile_original = tile.clone()
                
                # Pad if necessary
                tile_h, tile_w = tile.shape[2:]
                if tile_h < tile_size or tile_w < tile_size:
                    pad_h = tile_size - tile_h
                    pad_w = tile_size - tile_w
                    print(f"Padding: h+{pad_h}, w+{pad_w}")
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                
                # Process tile
                tile = tile.to(self.device)
                with torch.no_grad():
                    tile_output = self.model(tile)
                
                # Model already includes residual connection (input + residual)
                # So output is already the starless image, just clamp it
                print(f"Model output (raw) range: [{tile_output.min():.4f}, {tile_output.max():.4f}]")
                
                # Clamp to valid range [0, 1]
                tile_output = torch.clamp(tile_output, 0, 1)
                
                print(f"Tile output shape: {tile_output.shape}")
                print(f"Tile output range (after clamp): [{tile_output.min():.4f}, {tile_output.max():.4f}]")
                
                # Remove padding
                tile_output = tile_output[:, :, :tile_h, :tile_w]
                print(f"After unpad: {tile_output.shape}, range: [{tile_output.min():.4f}, {tile_output.max():.4f}]")
                
                # Apply weight and accumulate
                current_weight = weight_tile[:, :, :tile_h, :tile_w]
                print(f"Weight shape: {current_weight.shape}, range: [{current_weight.min():.4f}, {current_weight.max():.4f}]")
                output[:, :, i_start:i_end, j_start:j_end] += tile_output.cpu() * current_weight.cpu()
                weight[:, :, i_start:i_end, j_start:j_end] += current_weight.cpu()
        
        print("\n=== Finalization ===")
        print(f"Output before normalization range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
        
        # Normalize by weights
        output = output / weight.clamp(min=1e-8)
        
        print(f"Output after normalization range: [{output.min():.4f}, {output.max():.4f}]")
        output_clamped = torch.clamp(output, 0, 1)
        print(f"Output after clamp range: [{output_clamped.min():.4f}, {output_clamped.max():.4f}]")
        
        return output_clamped
    
    def create_weight_matrix(self, tile_size, overlap):
        """
        Create smooth weight matrix for blending
        Uses cosine tapering at edges for seamless blending
        """
        weight = torch.ones(1, 3, tile_size, tile_size)
        
        if overlap == 0:
            return weight
        
        # Create 1D taper
        taper = torch.zeros(overlap)
        for i in range(overlap):
            taper[i] = 0.5 * (1 - np.cos(np.pi * i / overlap))
        
        # Apply taper to edges
        # Top edge
        weight[:, :, :overlap, :] *= taper.view(1, 1, overlap, 1)
        # Bottom edge
        weight[:, :, -overlap:, :] *= taper.flip(0).view(1, 1, overlap, 1)
        # Left edge
        weight[:, :, :, :overlap] *= taper.view(1, 1, 1, overlap)
        # Right edge
        weight[:, :, :, -overlap:] *= taper.flip(0).view(1, 1, 1, overlap)
        
        return weight
    
    def pad_to_multiple(self, img, multiple=32):
        """Pad image to be divisible by multiple"""
        _, _, h, w = img.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        
        return img
    
    def unpad(self, img, original_size):
        """Remove padding"""
        h, w = original_size
        return img[:, :, :h, :w]


def main():
    root = tk.Tk()
    app = StarRemovalGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
