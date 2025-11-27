import os
from argparse import Namespace
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import StarDataset
from model import NAFNetSmall
from utils import PerceptualLoss, reconstruct_from_residual, ssim, save_checkpoint, load_image, tensor_to_image


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def reduce_tensor(tensor: torch.Tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def create_dataloaders(args: Namespace, distributed: bool):
    train_set = StarDataset(args.data_root, split="train")
    val_set = StarDataset(args.data_root, split="val")

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def build_model(device: torch.device, distributed: bool):
    model = NAFNetSmall().to(device)
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    return model


def compute_loss(pred: torch.Tensor, target: torch.Tensor, perceptual: nn.Module):
    l1 = F.l1_loss(pred, target)
    ssim_loss = 1.0 - ssim(pred, target)
    p_loss = perceptual(pred, target)
    return l1 + ssim_loss + p_loss


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    perceptual: nn.Module,
    device: torch.device,
    distributed: bool,
):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            residual = model(inputs)
            outputs = reconstruct_from_residual(inputs, residual)
            loss = compute_loss(outputs, targets, perceptual)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.detach()

    total_batches = len(train_loader)
    epoch_loss = epoch_loss / total_batches
    epoch_loss = reduce_tensor(epoch_loss) if distributed else epoch_loss
    return epoch_loss.item()


def validate(model, val_loader, perceptual: nn.Module, device: torch.device, distributed: bool):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            residual = model(inputs)
            outputs = reconstruct_from_residual(inputs, residual)
            loss = compute_loss(outputs, targets, perceptual)
            total_loss += loss.detach() * inputs.size(0)
            total_count += inputs.size(0)
    if distributed:
        total = torch.tensor([total_loss.item(), total_count], device=device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        total_loss = total[0]
        total_count = total[1]
    mean_loss = total_loss / total_count
    return mean_loss.item()


def train_model(args: Namespace):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, distributed)
    perceptual = PerceptualLoss().to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=device.type == "cuda")

    train_loader, val_loader, train_sampler, _ = create_dataloaders(args, distributed)
    steps_per_epoch = len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    start_epoch = 0
    best_val = float("inf")
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        (model.module if isinstance(model, DDP) else model).load_state_dict(state_dict, strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and isinstance(scaler, GradScaler):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", best_val)
    if train_sampler is not None:
        train_sampler.set_epoch(start_epoch)

    checkpoint_name = "nafnet_star_removal.pth"
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, perceptual, device, distributed
        )
        val_loss = validate(model, val_loader, perceptual, device, distributed)
        if is_main_process():
            print(f"Epoch {epoch+1}/{args.epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            if (epoch + 1) % args.save_every == 0 or val_loss < best_val:
                state = {
                    "epoch": epoch + 1,
                    "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val": min(best_val, val_loss),
                }
                save_checkpoint(state, checkpoint_name)
                best_val = min(best_val, val_loss)

    if distributed:
        dist.destroy_process_group()


def load_model_from_checkpoint(model: nn.Module, checkpoint_path: Optional[str], map_location: torch.device):
    if checkpoint_path is None:
        return model
    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model


def infer(model_path: str, image_path: str, device: torch.device, output_path: Optional[str] = None):
    if model_path is None:
        raise ValueError("A trained checkpoint path is required for inference.")
    model = NAFNetSmall().to(device)
    model = load_model_from_checkpoint(model, model_path, device)
    model.eval()
    img_tensor = load_image(image_path, device)
    with torch.no_grad():
        residual = model(img_tensor)
        result = reconstruct_from_residual(img_tensor, residual)
    image = tensor_to_image(result)
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_starless{ext}"
    image.save(output_path)
    return output_path
