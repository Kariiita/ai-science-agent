"""Standalone evaluation script for depth-estimation checkpoints.

Usage:
    python scripts/evaluate.py --model scripts.dorn:dorn_baseline \
        --checkpoint model_snapshots/best_model.pth --data_dir data
"""
import argparse
import importlib
import os
import sys

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.nyu_depth_v2 import NYUDepthV2
from scripts.depth_metrics import compute_depth_metrics


def _load_model(spec, checkpoint_path, device):
    if ":" in spec:
        module_name, func_name = spec.split(":", 1)
    else:
        module_name, func_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    model = factory()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _colorize_depth(depth, min_d=0.5, max_d=10.0):
    """Convert a depth map to a colourised PIL image for visualisation."""
    d = np.clip(depth, min_d, max_d)
    norm = ((d - min_d) / (max_d - min_d) * 255).astype(np.uint8)
    return Image.fromarray(norm).apply_lut(_TURBO_LUT) if hasattr(Image, "apply_lut") else Image.fromarray(norm)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a depth-estimation checkpoint")
    parser.add_argument("--model", type=str, default="scripts.dorn:dorn_baseline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = _load_model(args.model, args.checkpoint, device)
    val_dataset = NYUDepthV2(root=args.data_dir, split="val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    os.makedirs("output", exist_ok=True)

    all_preds = []
    all_gts = []
    scene_metrics = {"near": [], "mid": [], "far": []}

    with torch.no_grad():
        for idx, (images, depths) in enumerate(val_loader):
            images, depths = images.to(device), depths.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu())
            all_gts.append(depths.cpu())

            # Per-scene breakdown by mean GT depth
            mean_depth = depths[depths > 0].mean().item()
            m = compute_depth_metrics(outputs.cpu(), depths.cpu())
            if mean_depth < 2.0:
                scene_metrics["near"].append(m)
            elif mean_depth < 5.0:
                scene_metrics["mid"].append(m)
            else:
                scene_metrics["far"].append(m)

            # Save visualisations for first 10 samples
            if idx < 10:
                pred_np = outputs[0, 0].cpu().numpy()
                gt_np = depths[0, 0].cpu().numpy()
                _save_vis(pred_np, gt_np, idx)

    preds = torch.cat(all_preds)
    gts = torch.cat(all_gts)
    overall = compute_depth_metrics(preds, gts)

    print("EVALUATION RESULT:")
    for key in ("AbsRel", "RMSE", "MAE", "delta1", "delta2", "delta3"):
        print(f"  val_{key}={overall[key]:.4f}")

    print("\nPer-scene breakdown (AbsRel / delta1):")
    for scene in ("near", "mid", "far"):
        lst = scene_metrics[scene]
        if lst:
            ar = np.mean([m["AbsRel"] for m in lst])
            d1 = np.mean([m["delta1"] for m in lst])
            print(f"  {scene:4s} ({len(lst):3d} samples): val_AbsRel={ar:.4f} val_delta1={d1:.4f}")


def _save_vis(pred, gt, idx):
    """Save grayscale depth visualisations."""
    for name, arr in (("pred", pred), ("gt", gt)):
        norm = np.clip(arr, 0.5, 10.0)
        norm = ((norm - 0.5) / 9.5 * 255).astype(np.uint8)
        Image.fromarray(norm).save(os.path.join("output", f"sample_{idx:04d}_{name}.png"))


if __name__ == "__main__":
    main()
