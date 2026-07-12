"""Convert the official NYU Depth v2 .mat file to train/val PNG splits.

Downloads from the official source produce nyu_depth_v2_labeled.mat
containing 1449 labelled images (1200 train + 654 test, but the standard
split uses 249 from the test set).

Usage:
    python scripts/convert_nyu_mat.py --mat nyu_depth_v2_labeled.mat --out data

Requires: scipy, h5py, numpy, Pillow
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image


def load_mat(path):
    """Load NYU .mat file, returning (images, depths, train_indices, test_indices)."""
    try:
        import scipy.io as sio
        mat = sio.loadmat(path)
        images = mat["images"]  # (H, W, 3, N) uint8
        depths = mat["depths"]  # (H, W, N) float64 in metres
        train_split = mat.get("trainIdx", mat.get("trainIndices", None))
        test_split = mat.get("testIdx", mat.get("testIndices", None))
    except NotImplementedError:
        # scipy.io can't read v7.3 MAT files, fall back to h5py
        import h5py
        f = h5py.File(path, "r")
        images = np.array(f["images"])  # (N, 3, W, H) in h5py
        depths = np.array(f["depths"])  # (N, W, H)
        images = images.transpose(3, 2, 1, 0) if images.ndim == 4 else images
        depths = depths.transpose(2, 1, 0) if depths.ndim == 3 else depths
        train_split = np.array(f.get("#refs#/trainIdx", f.get("trainIdx")))
        test_split = np.array(f.get("#refs#/testIdx", f.get("testIdx")))
        f.close()
    return images, depths, train_split, test_split


def save_split(images, depths, indices, out_dir, split_name):
    """Save a subset of images/depths to out_dir/split_name/{rgb,depth}/."""
    rgb_dir = os.path.join(out_dir, split_name, "rgb")
    depth_dir = os.path.join(out_dir, split_name, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    idx_arr = np.array(indices).flatten()
    # Handle 1-based MATLAB indices
    if idx_arr.min() >= 1:
        idx_arr = idx_arr - 1

    for i, mat_idx in enumerate(idx_arr):
        mat_idx = int(mat_idx)
        # RGB: images may be (H,W,3,N) or (N,H,W,3)
        if images.ndim == 4 and images.shape[-1] == images.shape[0] and images.shape[3] == 3:
            img = images[:, :, :, mat_idx]
        elif images.ndim == 4 and images.shape[1] == 3:
            img = images[mat_idx].transpose(1, 2, 0)
        elif images.ndim == 4:
            img = images[:, :, :, mat_idx]
        else:
            img = images[mat_idx]

        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        Image.fromarray(img).save(os.path.join(rgb_dir, f"{i+1:04d}.png"))

        # Depth
        if depths.ndim == 3 and depths.shape[-1] != images.shape[0]:
            d = depths[:, :, mat_idx]
        elif depths.ndim == 3:
            d = depths[mat_idx]
        else:
            d = depths[mat_idx]

        # Save as 16-bit PNG in millimetres
        d_mm = (d * 1000).clip(0, 65535).astype(np.uint16)
        Image.fromarray(d_mm).save(os.path.join(depth_dir, f"{i+1:04d}.png"))

    return len(idx_arr)


def main():
    parser = argparse.ArgumentParser(description="Convert NYU .mat to PNG splits")
    parser.add_argument("--mat", type=str, required=True,
                        help="Path to nyu_depth_v2_labeled.mat")
    parser.add_argument("--out", type=str, default="data",
                        help="Output directory")
    args = parser.parse_args()

    print(f"Loading {args.mat}...")
    images, depths, train_split, test_split = load_mat(args.mat)
    print(f"Loaded: images={images.shape}, depths={depths.shape}")

    # If no split info, use standard 1200/249
    if train_split is None:
        n = images.shape[-1] if images.ndim == 4 else images.shape[0]
        train_split = list(range(1, 1201))
        test_split = list(range(1201, n + 1))

    n_train = save_split(images, depths, train_split, args.out, "train")
    print(f"Saved {n_train} training samples to {args.out}/train/")
    n_val = save_split(images, depths, test_split, args.out, "val")
    print(f"Saved {n_val} validation samples to {args.out}/val/")
    print("Done!")


if __name__ == "__main__":
    main()
