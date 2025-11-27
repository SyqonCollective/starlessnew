"""
Professional training script for MSRF-NAFNet
Optimized for RTX 5090 with all advanced features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from collections import defaultdict
import numpy as np

from model import create_msrf_nafnet_s, create_msrf_nafnet_m, create_msrf_nafnet_l
from dataset import create_dataloaders
from losses import CombinedLoss
from utils import (
    AverageMeter, EMA, save_checkpoint, load_checkpoint,
    calculate_psnr, calculate_ssim, visualize_results
)


class Trainer:
    """Professional trainer with all modern features"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.vis_dir = self.output_dir / 'visualizations'
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.vis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Build model
        self.build_model()
        
        # Setup data loaders
        self.setup_dataloaders()
        
        # Setup loss
        self.criterion = CombinedLoss(
            l1_weight=config['loss']['l1_weight'],
            perceptual_weight=config['loss']['perceptual_weight'],
            texture_weight=config['loss']['texture_weight'],
            edge_weight=config['loss']['edge_weight'],
            frequency_weight=config['loss']['frequency_weight']
        ).to(self.device)
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Setup scheduler
        self.setup_scheduler()
        
        # Mixed precision training
        self.use_amp = config['training']['use_amp']
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # EMA for model weights
        self.use_ema = config['training']['use_ema']
        if self.use_ema:
            self.ema = EMA(self.model, decay=config['training']['ema_decay'])
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_psnr = 0.0
        
        # Gradient accumulation
        self.accumulation_steps = config['training']['gradient_accumulation_steps']
    
    def build_model(self):
        """Build model architecture"""
        model_type = self.config['model']['type']
        use_grad_checkpoint = self.config['training'].get('use_gradient_checkpointing', False)
        
        if model_type == 'msrf_nafnet_s':
            self.model = create_msrf_nafnet_s(use_gradient_checkpointing=use_grad_checkpoint)
        elif model_type == 'msrf_nafnet_m':
            self.model = create_msrf_nafnet_m(use_gradient_checkpointing=use_grad_checkpoint)
        elif model_type == 'msrf_nafnet_l':
            self.model = create_msrf_nafnet_l(use_gradient_checkpointing=use_grad_checkpoint)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nModel: {model_type}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Compile model for PyTorch 2.0+ (massive speedup on RTX 5090)
        if hasattr(torch, 'compile') and self.config['training'].get('compile_model', True):
            print("Compiling model with torch.compile for optimal performance...")
            self.model = torch.compile(self.model, mode='max-autotune')
    
    def setup_dataloaders(self):
        """Setup data loaders"""
        self.train_loader, self.val_loader = create_dataloaders(
            root_dir=self.config['data']['root_dir'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            patch_size=self.config['data']['patch_size'],
            pin_memory=True,
            prefetch=True,
            device=self.device,
            subset_fraction=self.config['data'].get('subset_fraction', 1.0)  # Usa subset se specificato
        )
        
        print(f"\nDataset loaded:")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
    
    def setup_optimizer(self):
        """Setup optimizer"""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=(opt_config['beta1'], opt_config['beta2'])
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        sched_config = self.config['scheduler']
        
        if sched_config['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config['min_lr']
            )
        elif sched_config['type'] == 'cosine_warmup':
            from utils import CosineAnnealingWarmupRestarts
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                first_cycle_steps=sched_config['first_cycle_steps'],
                cycle_mult=sched_config.get('cycle_mult', 1.0),
                max_lr=self.config['optimizer']['lr'],
                min_lr=sched_config['min_lr'],
                warmup_steps=sched_config['warmup_steps'],
                gamma=sched_config.get('gamma', 1.0)
            )
        elif sched_config['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)  # Libera memoria
        
        losses = AverageMeter()
        psnr_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['epochs']}")
        
        for i, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights with gradient accumulation
            if (i + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # Update EMA
                if self.use_ema:
                    self.ema.update()
                
                self.global_step += 1
            
            # Calculate metrics
            with torch.no_grad():
                psnr = calculate_psnr(outputs, targets)
            
            # Update meters
            losses.update(loss.item() * self.accumulation_steps, inputs.size(0))
            psnr_meter.update(psnr, inputs.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'psnr': f'{psnr_meter.avg:.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to tensorboard
            if self.global_step % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', losses.avg, self.global_step)
                self.writer.add_scalar('train/psnr', psnr_meter.avg, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        return losses.avg, psnr_meter.avg
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate model"""
        # Use EMA model for validation if available
        if self.use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        losses = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for i, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Calculate metrics
            psnr = calculate_psnr(outputs, targets)
            ssim = calculate_ssim(outputs, targets)
            
            # Update meters
            losses.update(loss.item(), inputs.size(0))
            psnr_meter.update(psnr, inputs.size(0))
            ssim_meter.update(ssim, inputs.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'psnr': f'{psnr_meter.avg:.2f}',
                'ssim': f'{ssim_meter.avg:.4f}'
            })
            
            # Visualize first batch
            if i == 0:
                vis_path = self.vis_dir / f'epoch_{epoch:04d}.png'
                visualize_results(inputs, outputs, targets, vis_path, max_images=4)
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', losses.avg, epoch)
        self.writer.add_scalar('val/psnr', psnr_meter.avg, epoch)
        self.writer.add_scalar('val/ssim', ssim_meter.avg, epoch)
        
        # Restore original model weights
        if self.use_ema:
            self.ema.restore()
        
        return losses.avg, psnr_meter.avg, ssim_meter.avg
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_psnr = self.train_epoch(epoch)
            
            # Validate
            if (epoch + 1) % self.config['training']['val_interval'] == 0:
                val_loss, val_psnr, val_ssim = self.validate(epoch)
                
                # Save best model
                if val_psnr > self.best_psnr:
                    self.best_psnr = val_psnr
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        self.best_psnr,
                        self.checkpoint_dir / 'best_model.pth',
                        ema=self.ema if self.use_ema else None
                    )
                    print(f"\n✓ New best model saved! PSNR: {val_psnr:.2f} dB")
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.best_psnr,
                    self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth',
                    ema=self.ema if self.use_ema else None
                )
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Final save
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.current_epoch,
            self.best_psnr,
            self.checkpoint_dir / 'final_model.pth',
            ema=self.ema if self.use_ema else None
        )
        
        total_time = time.time() - start_time
        print(f"\n" + "="*50)
        print(f"Training completed in {total_time/3600:.2f} hours")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print("="*50)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train MSRF-NAFNet for star removal')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume if checkpoint provided
    if args.resume:
        load_checkpoint(
            args.resume,
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            ema=trainer.ema if trainer.use_ema else None
        )
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
