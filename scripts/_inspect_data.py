"""Data Agent inspection script - produces DATA_REPORT.md and DATASET_MANIFEST.json."""
import os, json, glob
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

DATA_ROOT = Path(r"D:\code\ai-science-agent-team\depth_project\data")
WORKSPACE = Path(r"D:\code\ai-science-agent-team\depth_project\workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)

manifest = {"version": "1.0", "datasets": {}, "total_trainable": {"train": 0, "val": 0}}
report_lines = ["# Data Analysis Report", f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]

for split in ["train", "val"]:
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        continue
    
    rgb_files = sorted(split_dir.glob("*_colors.png"))
    depth_files = sorted(split_dir.glob("*_depth.png"))
    
    pairs = []
    for rgb in rgb_files:
        stem = rgb.name.replace("_colors.png", "")
        depth = split_dir / f"{stem}_depth.png"
        if depth.exists():
            pairs.append((rgb, depth))
    
    stats = {"count": len(pairs), "rgb_format": "png", "depth_format": "png16"}
    
    if pairs:
        rgb_sample = np.array(Image.open(pairs[0][0]))
        depth_sample = np.array(Image.open(pairs[0][1]), dtype=np.float32)
        depth_meters = depth_sample / 1000.0
        
        stats["resolution"] = list(rgb_sample.shape)
        stats["channels"] = rgb_sample.shape[2] if len(rgb_sample.shape) == 3 else 1
        stats["depth_resolution"] = list(depth_sample.shape)
        stats["depth_min_m"] = float(depth_meters[depth_meters > 0].min())
        stats["depth_max_m"] = float(depth_meters.max())
        stats["depth_mean_m"] = float(depth_meters[depth_meters > 0].mean())
        stats["depth_median_m"] = float(np.median(depth_meters[depth_meters > 0]))
        stats["depth_std_m"] = float(depth_meters[depth_meters > 0].std())
        stats["invalid_pixels_pct"] = float((depth_meters == 0).sum() / depth_meters.size * 100)
        
        valid = depth_meters[depth_meters > 0]
        hist, edges = np.histogram(valid, bins=10)
        stats["histogram"] = {"counts": hist.tolist(), "edges": [round(e, 2) for e in edges.tolist()]}
        
        near = (valid < 2.0).sum()
        far = (valid > 5.0).sum()
        stats["near_field_pct"] = float(near / len(valid) * 100)
        stats["far_field_pct"] = float(far / len(valid) * 100)
    
    manifest["datasets"][split] = stats
    manifest["total_trainable"][split] = len(pairs)

manifest_path = WORKSPACE / "DATASET_MANIFEST.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

report_lines.append("## Dataset Overview")
for split, s in manifest["datasets"].items():
    report_lines.append(f"- {split}: {s['count']} samples")
    report_lines.append(f"  - RGB: {s.get('rgb_format','?')}, {s.get('resolution','?')}, {s.get('channels','?')} channels")
    report_lines.append(f"  - Depth: {s.get('depth_format','?')}, {s.get('depth_resolution','?')}")
report_lines.append("")

report_lines.append("## Depth Distribution")
report_lines.append("| Metric | Train | Val |")
report_lines.append("|--------|-------|-----|")
for metric in ["depth_min_m", "depth_max_m", "depth_mean_m", "depth_median_m", "depth_std_m", "invalid_pixels_pct"]:
    train_val = manifest["datasets"].get("train", {}).get(metric, "N/A")
    val_val = manifest["datasets"].get("val", {}).get(metric, "N/A")
    label = metric.replace("depth_", "").replace("_m", " (m)").replace("_pct", " (%)").replace("_", " ").title()
    if isinstance(train_val, float):
        report_lines.append(f"| {label} | {train_val:.3f} | {val_val:.3f} |")
    else:
        report_lines.append(f"| {label} | {train_val} | {val_val} |")
report_lines.append("")

report_lines.append("## Data Quality Issues")
train = manifest["datasets"].get("train", {})
if train.get("invalid_pixels_pct", 0) > 0:
    report_lines.append(f"1. Depth holes: {train['invalid_pixels_pct']:.1f}% of pixels are invalid (0 depth) in train set")
report_lines.append("")

report_lines.append("## Recommendations for Next Experiment")
report_lines.append("1. Handle invalid depth pixels (mask out zero-depth regions in loss function)")
report_lines.append("2. Consider depth range normalization (clip to 0-10m range based on distribution)")
report_lines.append("3. Data augmentation: random crops and color jitter for RGB images")
report_lines.append("")

report_path = WORKSPACE / "DATA_REPORT.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("=== DATA_REPORT.md ===")
print("\n".join(report_lines))
print("\n=== DATASET_MANIFEST.json ===")
print(json.dumps(manifest, indent=2))