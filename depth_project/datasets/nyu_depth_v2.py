"""NYU Depth v2 dataset loader.

Supports the standard directory layout:
    root/
      train/  rgb/ *.png   depth/ *.png
      val/    rgb/ *.png   depth/ *.png

Depth formats: 16-bit PNG (millimetres or metres), .npy, .tiff, .exr.
RGB is resized, converted to tensor, and ImageNet-normalised.
Depth is resized with nearest-neighbour and clipped to [0.5, 10.0] metres.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_RGB_MEAN = [0.485, 0.456, 0.406]
_RGB_STD = [0.229, 0.224, 0.225]
_DEFAULT_SIZE = (256, 256)
_DEPTH_MIN = 0.5
_DEPTH_MAX = 10.0
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _natural_key(name: str) -> tuple:
    return tuple(int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", name))


def _read_depth(path: Path) -> np.ndarray:
    """Read a depth map and return float32 in metres, shape (H, W)."""
    ext = path.suffix.lower()
    if ext == ".npy":
        depth = np.load(str(path)).astype(np.float32)
    elif ext == ".exr":
        import cv2
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    elif ext in (".tif", ".tiff"):
        depth = np.array(Image.open(str(path)), dtype=np.float32)
    else:
        raw = np.array(Image.open(str(path)), dtype=np.float32)
        depth = raw
        if depth.max() > 100.0:
            depth = depth / 1000.0
    return depth


def _resize_depth(depth: np.ndarray, size: tuple) -> np.ndarray:
    """Resize depth with nearest-neighbour to avoid phantom depth values."""
    img = Image.fromarray(depth.astype(np.float32), mode="F")
    img = img.resize(size, Image.NEAREST)
    return np.array(img, dtype=np.float32)


class NYUDepthV2(Dataset):
    """NYU Depth v2 labelled-subset dataset."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
        size: tuple = _DEFAULT_SIZE,
    ):
        self.root = Path(root)
        self.split = split
        self.size = size

        rgb_dir = self.root / split / "rgb"
        depth_dir = self.root / split / "depth"

        if not rgb_dir.is_dir():
            raise FileNotFoundError(
                f"RGB directory not found: {rgb_dir}\n"
                f"Expected layout: {self.root}/{{train,val}}/{{rgb,depth}}/"
            )

        self.samples: list[tuple[Path, Path]] = []
        rgb_files = sorted(
            [f for f in rgb_dir.iterdir() if f.suffix.lower() in _IMG_EXTS],
            key=lambda p: _natural_key(p.name),
        )
        for rgb_path in rgb_files:
            stem = rgb_path.stem
            candidates = [
                depth_dir / f"{stem}.png",
                depth_dir / f"{stem}.npy",
                depth_dir / f"{stem}.tiff",
                depth_dir / f"{stem}.exr",
                depth_dir / rgb_path.name,
            ]
            depth_path = next((c for c in candidates if c.exists()), None)
            if depth_path is None:
                raise FileNotFoundError(
                    f"No depth map for RGB image {rgb_path.name} in {depth_dir}"
                )
            self.samples.append((rgb_path, depth_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No RGB images found in {rgb_dir}")

        if transform is not None:
            self.rgb_transform = transform
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=_RGB_MEAN, std=_RGB_STD),
            ])
        self.depth_transform = depth_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_path, depth_path = self.samples[idx]
        rgb = Image.open(str(rgb_path)).convert("RGB")
        image = self.rgb_transform(rgb)
        depth = _read_depth(depth_path)
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)
        depth = _resize_depth(depth, self.size)
        depth = np.clip(depth, _DEPTH_MIN, _DEPTH_MAX)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        return image, depth_tensor

    def __repr__(self) -> str:
        return (
            f"NYUDepthV2(root={self.root}, split={self.split}, "
            f"size={self.size}, samples={len(self.samples)})"
        )
