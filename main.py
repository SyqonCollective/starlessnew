import argparse
import os
import random

import numpy as np
import torch

from train import train_model, infer


def parse_args():
    parser = argparse.ArgumentParser(description="NAFNet-S star removal training")
    parser.add_argument("--data_root", type=str, default=".", help="Root directory containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to resume or infer")
    parser.add_argument("--infer_image", type=str, default=None, help="Run inference on a single image and exit")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for inference")
    return parser.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.infer_image is not None:
        output_path = infer(args.checkpoint, args.infer_image, device, args.output)
        print(f"Saved starless image to {output_path}")
        return

    if not os.path.isdir(os.path.join(args.data_root, "train")) or not os.path.isdir(
        os.path.join(args.data_root, "val")
    ):
        raise FileNotFoundError("Expected train/ and val/ directories inside data_root")

    train_model(args)


if __name__ == "__main__":
    main()
