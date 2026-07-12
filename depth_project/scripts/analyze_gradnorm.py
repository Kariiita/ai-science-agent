import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
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
from tools.gradnorm_loss import gradnorm_loss


def analyze_residuals_and_gradients(model, val_loader, device, num_samples=10):
    """Analyze gradient magnitude distribution and depth map residuals.
    
    Computes:
    - Per-depth-bin RMSE (0-1m, 1-3m, 3-6m, 6-10m)
    - Residual heatmaps at occlusion boundaries
    - Gradient magnitude statistics
    """
    model.eval()
    all_preds = []
    all_gts = []
    
    # Depth bins
    bins = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 10.0)]
    bin_rmse = [0.0 for _ in bins]
    bin_counts = [0 for _ in bins]
    
    # For residual heatmap analysis
    residuals_list = []
    
    with torch.no_grad():
        for i, (images, depths) in enumerate(val_loader):
            if i >= num_samples:
                break
            images, depths = images.to(device), depths.to(device)
            outputs = model(images)
            
            # Store predictions and ground truth
            all_preds.append(outputs.cpu())
            all_gts.append(depths.cpu())
            
            # Compute residuals
            residuals = (outputs - depths).cpu().numpy()  # (B, 1, H, W)
            residuals_list.append(residuals)
            
            # Compute per-pixel depth bin assignment
            gt_np = depths.cpu().numpy()
            for j, (low, high) in enumerate(bins):
                mask = (gt_np >= low) & (gt_np < high)
                if mask.sum() > 0:
                    bin_rmse[j] += ((residuals[mask]) ** 2).sum()
                    bin_counts[j] += mask.sum()
            
            # Compute gradient magnitudes
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                                 dtype=torch.float32, device=outputs.device)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                                 dtype=torch.float32, device=outputs.device)
            
            dx = F.conv2d(outputs, sobel_x, padding=1)
            dy = F.conv2d(outputs, sobel_y, padding=1)
            
            grad_mag = torch.sqrt(dx**2 + dy**2 + 1e-8)
            
            # Print gradient stats for this batch
            print(f'Batch {i+1}: grad_mag mean={grad_mag.mean().item():.4f}, std={grad_mag.std().item():.4f}')
    
    # Compute final RMSE per bin
    bin_rmse_final = []
    for j in range(len(bins)):
        if bin_counts[j] > 0:
            bin_rmse_final.append(np.sqrt(bin_rmse[j] / bin_counts[j]))
        else:
            bin_rmse_final.append(0.0)
    
    # Overall metrics
    preds = torch.cat(all_preds)
    gts = torch.cat(all_gts)
    overall_metrics = compute_depth_metrics(preds, gts)
    
    # Print results
    print('\n=== GRADNORM ANALYSIS RESULTS ===')
    print('Per-depth-bin RMSE:')
    for j, (low, high) in enumerate(bins):
        print(f'  {low:.1f}-{high:.1f}m: {bin_rmse_final[j]:.4f}')
    print('\nOverall metrics:')
    for key, value in overall_metrics.items():
        print(f'  {key}: {value:.4f}')
    
    # Save residual heatmaps for first sample
    if residuals_list:
        os.makedirs('output/gradnorm_analysis', exist_ok=True)
        
        # Visualize first residual map
        res = residuals_list[0][0, 0]  # (H, W)
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(res, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
        plt.colorbar()
        plt.title('Residual Heatmap (first sample)')
        
        plt.subplot(1, 2, 2)
        plt.hist(res.flatten(), bins=50, alpha=0.7, label='Residuals')
        plt.xlabel('Residual (m)')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('output/gradnorm_analysis/residual_analysis.png')
        plt.close()
        
        print('\nSaved residual analysis plot to output/gradnorm_analysis/residual_analysis.png')
    
    return bin_rmse_final, overall_metrics


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
    bin_rmse, overall_metrics = analyze_residuals_and_gradients(model, val_loader, device)
    
    print('\nAnalysis completed!')
    import os
    os._exit(0)


if __name__ == '__main__':
    main()
