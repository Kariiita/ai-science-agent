import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.nyu_depth_v2 import NYUDepthV2
from torch.utils.data import DataLoader
from scripts.train_gradnorm import _load_model
from scripts.depth_metrics import compute_depth_metrics


def analyze_residuals_and_gradients(model, val_loader, device, num_samples=None):
    """Analyze gradient magnitude distribution and depth map residuals.
    
    Computes:
    - Per-depth-bin RMSE (0-10m, 10-20m, 20-30m, 30+ m)
    - Mean absolute residual and std residual per bin
    - Residual histogram
    """
    model.eval()
    all_preds = []
    all_gts = []
    
    # Depth bins: 0–10m, 10–20m, 20–30m, 30+ m
    bins = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, float('inf'))]
    bin_rmse = [0.0 for _ in bins]
    bin_abs_residual = [0.0 for _ in bins]
    bin_std_residual = [0.0 for _ in bins]
    bin_counts = [0 for _ in bins]
    
    # For residual histogram
    all_residuals = []
    
    with torch.no_grad():
        for i, (images, depths) in enumerate(val_loader):
            if num_samples is not None and i >= num_samples:
                break
            images, depths = images.to(device), depths.to(device)
            outputs = model(images)
            
            # Store predictions and ground truth
            all_preds.append(outputs.cpu())
            all_gts.append(depths.cpu())
            
            # Compute residuals
            residuals = (outputs - depths).cpu().numpy()  # (B, 1, H, W)
            all_residuals.append(residuals)
            
            # Compute per-pixel depth bin assignment
            gt_np = depths.cpu().numpy()
            for j, (low, high) in enumerate(bins):
                mask = (gt_np >= low) & (gt_np < high)
                if mask.sum() > 0:
                    residuals_masked = residuals[mask]
                    bin_rmse[j] += ((residuals_masked) ** 2).sum()
                    bin_abs_residual[j] += np.abs(residuals_masked).sum()
                    bin_std_residual[j] += np.std(residuals_masked) * len(residuals_masked)
                    bin_counts[j] += mask.sum()
    
    # Compute final metrics per bin
    bin_rmse_final = []
    bin_abs_residual_final = []
    bin_std_residual_final = []
    for j in range(len(bins)):
        if bin_counts[j] > 0:
            bin_rmse_final.append(np.sqrt(bin_rmse[j] / bin_counts[j]))
            bin_abs_residual_final.append(bin_abs_residual[j] / bin_counts[j])
            bin_std_residual_final.append(np.sqrt(bin_std_residual[j] / bin_counts[j]))
        else:
            bin_rmse_final.append(0.0)
            bin_abs_residual_final.append(0.0)
            bin_std_residual_final.append(0.0)
    
    # Overall metrics
    preds = torch.cat(all_preds)
    gts = torch.cat(all_gts)
    overall_metrics = compute_depth_metrics(preds, gts)
    
    # Concatenate all residuals for histogram
    all_residuals = np.concatenate([r for r in all_residuals], axis=0).flatten()
    
    # Prepare output dict
    result = {
        "bins": [
            {"range": "0–10m", "rmse": bin_rmse_final[0], "mean_abs_residual": bin_abs_residual_final[0], "std_residual": bin_std_residual_final[0]},
            {"range": "10–20m", "rmse": bin_rmse_final[1], "mean_abs_residual": bin_abs_residual_final[1], "std_residual": bin_std_residual_final[1]},
            {"range": "20–30m", "rmse": bin_rmse_final[2], "mean_abs_residual": bin_abs_residual_final[2], "std_residual": bin_std_residual_final[2]},
            {"range": "30+ m", "rmse": bin_rmse_final[3], "mean_abs_residual": bin_abs_residual_final[3], "std_residual": bin_std_residual_final[3]}
        ],
        "overall_metrics": overall_metrics,
        "residual_histogram": {
            "min": float(np.min(all_residuals)),
            "max": float(np.max(all_residuals)),
            "mean": float(np.mean(all_residuals)),
            "std": float(np.std(all_residuals)),
            "count": len(all_residuals)
        }
    }
    
    return result


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint_path = 'model_snapshots/checkpoint_epoch_50.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print(f'Loaded checkpoint from {checkpoint_path}')
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
        raise
    
    # Load model
    model = _load_model('scripts.dorn:dorn_baseline').to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    data_dir = 'data'
    try:
        val_dataset = NYUDepthV2(root=data_dir, split='val')
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f'Loaded {len(val_dataset)} validation samples')
    except Exception as e:
        print(f'Error loading dataset: {e}')
        raise
    
    # Run analysis
    result = analyze_residuals_and_gradients(model, val_loader, device)
    
    # Save results to logs/residual_analysis_gradnorm_epoch50.json
    os.makedirs('logs', exist_ok=True)
    import json
    with open('logs/residual_analysis_gradnorm_epoch50.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print('\n=== GRADNORM RESIDUAL ANALYSIS RESULTS ===')
    print('Per-depth-bin metrics:')
    for bin_data in result["bins"]:
        print(f'  {bin_data["range"]}: RMSE={bin_data["rmse"]:.4f}, MeanAbsRes={bin_data["mean_abs_residual"]:.4f}, StdRes={bin_data["std_residual"]:.4f}')
    print('\nOverall metrics:')
    for key, value in result["overall_metrics"].items():
        print(f'  {key}: {value:.4f}')
    print('\nResidual histogram stats:')
    hist = result["residual_histogram"]
    print(f'  Min: {hist["min"]:.4f}, Max: {hist["max"]:.4f}')
    print(f'  Mean: {hist["mean"]:.4f}, Std: {hist["std"]:.4f}')
    print(f'  Count: {hist["count"]}')
    print('\nSaved results to logs/residual_analysis_gradnorm_epoch50.json')
    print('\nAnalysis completed!')
    os._exit(0)


if __name__ == '__main__':
    main()
