"""Download NYU Depth v2 labelled .mat from HuggingFace mirror and convert to PNG splits.

Usage:
    python scripts/download_nyu_hf.py --out data

The official .mat file (~2.8GB) is downloaded once, then converted to
train/val PNG splits matching the directory layout expected by NYUDepthV2.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def download_mat(repo_id, filename, cache_dir):
    from huggingface_hub import hf_hub_download
    print("Downloading %s from %s ..." % (filename, repo_id))
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    print("Downloaded to:", path)
    return path


def load_mat(path):
    try:
        import scipy.io as sio
        mat = sio.loadmat(path)
        images = mat["images"]
        depths = mat["depths"]
        train_idx = mat.get("trainIdx", mat.get("trainIndices"))
        test_idx = mat.get("testIdx", mat.get("testIndices"))
        return images, depths, train_idx, test_idx
    except NotImplementedError:
        import h5py
        f = h5py.File(path, "r")
        images = np.array(f["images"])
        depths = np.array(f["depths"])
        # h5py stores MATLAB arrays transposed
        if images.ndim == 4 and images.shape[0] == 3:
            images = images.transpose(2, 1, 0, 3) if images.shape[3] < images.shape[2] else images.transpose(3, 2, 1, 0)
        if depths.ndim == 3:
            depths = depths.transpose(2, 1, 0)
        train_idx = np.array(f["trainIdx"]) if "trainIdx" in f else None
        test_idx = np.array(f["testIdx"]) if "testIdx" in f else None
        f.close()
        return images, depths, train_idx, test_idx


def save_split(images, depths, indices, out_dir, split_name):
    rgb_dir = os.path.join(out_dir, split_name, "rgb")
    depth_dir = os.path.join(out_dir, split_name, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    idx_arr = np.array(indices).flatten()
    if len(idx_arr) > 0 and idx_arr.min() >= 1:
        idx_arr = idx_arr - 1

    count = 0
    for i, mat_idx in enumerate(idx_arr):
        mat_idx = int(mat_idx)
        if images.ndim == 4 and images.shape[-1] != 3 and images.shape[1] == 3:
            img = images[mat_idx].transpose(1, 2, 0)
        elif images.ndim == 4:
            img = images[:, :, :, mat_idx]
        else:
            img = images[mat_idx]
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        Image.fromarray(img).save(os.path.join(rgb_dir, "%04d.png" % (i + 1)))

        if depths.ndim == 3 and depths.shape[-1] != 3:
            d = depths[:, :, mat_idx]
        elif depths.ndim == 3:
            d = depths[mat_idx]
        else:
            d = depths[mat_idx]
        d_mm = (d * 1000).clip(0, 65535).astype(np.uint16)
        Image.fromarray(d_mm).save(os.path.join(depth_dir, "%04d.png" % (i + 1)))
        count += 1
        if count % 100 == 0:
            print("  %s: %d/%d" % (split_name, count, len(idx_arr)))
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--repo", type=str, default="sayakpaul/nyu_depth_v2")
    parser.add_argument("--filename", type=str, default="nyu_depth_v2_labeled.mat")
    args = parser.parse_args()

    cache_dir = os.path.join(os.path.dirname(args.out), ".hf_cache")
    mat_path = download_mat(args.repo, args.filename, cache_dir)

    print("Loading .mat file...")
    images, depths, train_idx, test_idx = load_mat(mat_path)
    print("Images shape:", images.shape, "Depths shape:", depths.shape)

    if train_idx is None:
        n = images.shape[-1] if images.ndim == 4 else images.shape[0]
        train_idx = list(range(1, 1201))
        test_idx = list(range(1201, n + 1))

    n_train = save_split(images, depths, train_idx, args.out, "train")
    print("Train: %d samples" % n_train)
    n_val = save_split(images, depths, test_idx, args.out, "val")
    print("Val: %d samples" % n_val)
    print("Done! Data saved to:", args.out)


if __name__ == "__main__":
    main()
