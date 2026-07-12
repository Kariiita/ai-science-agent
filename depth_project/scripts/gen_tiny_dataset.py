"""Generate a small synthetic NYU Depth v2-like dataset for pipeline testing.

Creates simple RGB images with geometric depth patterns so the DORN model
has something learnable to fit.

Usage:
    python scripts/gen_tiny_dataset.py --out data --n_train 80 --n_val 20
"""
import argparse
import os
import numpy as np
from PIL import Image


def _make_sample(idx, size=(256, 256)):
    w, h = size
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    np.random.seed(idx)
    angle = np.random.uniform(0, 2 * np.pi)
    freq = np.random.uniform(0.5, 2.0)
    offset = np.random.uniform(2.0, 5.0)
    gx = xx * np.cos(angle) + yy * np.sin(angle)
    depth = offset + freq * gx / w + 0.5 * np.sin(xx / 30.0) * np.cos(yy / 40.0)
    depth = np.clip(depth, 0.5, 10.0).astype(np.float32)
    for _ in range(np.random.randint(2, 5)):
        cx = np.random.randint(30, w - 30)
        cy = np.random.randint(30, h - 30)
        rw = np.random.randint(10, 40)
        rh = np.random.randint(10, 40)
        d_val = np.random.uniform(1.0, 8.0)
        depth[max(0, cy-rh):cy+rh, max(0, cx-rw):cx+rw] = d_val
    tint = np.random.uniform(0.7, 1.3, size=3)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        rgb[:, :, c] = np.clip(255 * (1.0 - depth / 12.0) * tint[c], 0, 255)
    rgb = rgb.astype(np.uint8)
    return rgb, depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_val", type=int, default=20)
    args = parser.parse_args()

    for split_name, n_samples in [("train", args.n_train), ("val", args.n_val)]:
        rgb_dir = os.path.join(args.out, split_name, "rgb")
        depth_dir = os.path.join(args.out, split_name, "depth")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        base = 0 if split_name == "train" else 10000
        for i in range(n_samples):
            idx = base + i
            rgb, depth = _make_sample(idx)
            Image.fromarray(rgb).save(os.path.join(rgb_dir, "%04d.png" % (i + 1)))
            depth_mm = (depth * 1000).clip(0, 65535).astype(np.uint16)
            Image.fromarray(depth_mm).save(os.path.join(depth_dir, "%04d.png" % (i + 1)))
        print("%s: %d samples saved" % (split_name, n_samples))
    print("Done! Synthetic data saved to:", args.out)


if __name__ == "__main__":
    main()
