import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class StarDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.root = root
        self.split = split
        self.samples = self._collect_pairs()

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        pairs = []
        for fname in os.listdir(split_dir):
            if not fname.endswith("_input.png"):
                continue
            base = fname[:-10]  # remove _input.png
            input_path = os.path.join(split_dir, fname)
            target_name = f"{base}_target.png"
            target_path = os.path.join(split_dir, target_name)
            if os.path.isfile(target_path):
                pairs.append((input_path, target_path))

        if not pairs:
            raise RuntimeError(f"No input/target pairs found in {split_dir}")
        pairs.sort()
        return pairs

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor

    def __getitem__(self, idx: int):
        input_path, target_path = self.samples[idx]
        inp = self._load_image(input_path)
        target = self._load_image(target_path)
        return inp, target
