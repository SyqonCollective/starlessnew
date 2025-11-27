"""
Utility functions for training and evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional
import math


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply EMA weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warmup and restarts
    """
    def __init__(
        self,
        optimizer,
        first_cycle_steps,
        cycle_mult=1.0,
        max_lr=0.1,
        min_lr=0.001,
        warmup_steps=0,
        gamma=1.0,
        last_epoch=-1
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # Set learning rate
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr)
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def calculate_psnr(pred, target, max_val=1.0):
    """Calculate PSNR between prediction and target"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def calculate_ssim(pred, target, window_size=11, max_val=1.0):
    """Calculate SSIM between prediction and target"""
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    pred = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)
    
    # Create window
    window = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
    window = window.to(pred.device)
    
    mu1 = torch.nn.functional.conv2d(pred, window, padding=window_size//2, groups=pred.shape[1])
    mu2 = torch.nn.functional.conv2d(target, window, padding=window_size//2, groups=target.shape[1])
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.conv2d(pred * pred, window, padding=window_size//2, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target * target, window, padding=window_size//2, groups=target.shape[1]) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred * target, window, padding=window_size//2, groups=pred.shape[1]) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, path, ema=None):
    """Save training checkpoint"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }
    
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    
    if ema is not None:
        state['ema_state_dict'] = ema.state_dict()
    
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, ema=None):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_metric', 0)


def visualize_results(inputs, outputs, targets, save_path, max_images=4):
    """Visualize and save results"""
    n = min(max_images, inputs.size(0))
    
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        # Convert to numpy and transpose
        inp = inputs[i].cpu().permute(1, 2, 0).numpy()
        out = outputs[i].cpu().permute(1, 2, 0).numpy()
        tgt = targets[i].cpu().permute(1, 2, 0).numpy()
        
        # Clamp to [0, 1]
        inp = np.clip(inp, 0, 1)
        out = np.clip(out, 0, 1)
        tgt = np.clip(tgt, 0, 1)
        
        # Plot
        axes[i, 0].imshow(inp)
        axes[i, 0].set_title('Input (with stars)')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(out)
        axes[i, 1].set_title('Output (stars removed)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(tgt)
        axes[i, 2].set_title('Target (ground truth)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
