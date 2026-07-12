import torch
import torch.nn as nn
import torch.nn.functional as F


def depth_binned_l1_loss_debug(pred, gt, bins=None, bin_weights=None, scale_by_median=True, eps=1e-6):
    """
    Debug version: prints all intermediate values.
    """
    if bins is None:
        bins = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    
    print(f'[DEBUG] pred shape: {pred.shape}, gt shape: {gt.shape}')
    print(f'[DEBUG] pred range: {pred.min().item():.4f}–{pred.max().item():.4f}')
    print(f'[DEBUG] gt range: {gt.min().item():.4f}–{gt.max().item():.4f}')
    
    # Flatten predictions and ground truth
    pred_flat = pred.view(-1)
    gt_flat = gt.view(-1)
    
    # Mask valid pixels (gt > 0)
    valid_mask = gt_flat > 0
    print(f'[DEBUG] Valid mask sum: {valid_mask.sum().item()}')
    
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]
    
    if pred_flat.numel() == 0:
        print('[DEBUG] No valid pixels → returning 0')
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # Assign each pixel to a bin
    bin_assignments = torch.zeros_like(gt_flat, dtype=torch.long)
    for i in range(len(bins) - 1):
        mask = (gt_flat >= bins[i]) & (gt_flat < bins[i+1])
        bin_assignments[mask] = i
        print(f'[DEBUG] Bin {i} ({bins[i]}–{bins[i+1]}): {(mask).sum().item()} pixels')
    
    # Compute bin frequencies
    bin_counts = torch.zeros(len(bins)-1, device=pred.device)
    for i in range(len(bins)-1):
        bin_counts[i] = (bin_assignments == i).sum().float()
        print(f'[DEBUG] Bin {i} count: {bin_counts[i].item()}')
    
    # Compute bin weights: inverse frequency normalized to sum to 1
    if bin_weights is None:
        bin_weights = 1.0 / (bin_counts + eps)
        bin_weights = bin_weights / bin_weights.sum()
        print(f'[DEBUG] Bin weights: {bin_weights.tolist()}')
    
    # Compute bin medians for scaling
    bin_medians = torch.zeros(len(bins)-1, device=pred.device)
    for i in range(len(bins)-1):
        bin_mask = bin_assignments == i
        if bin_mask.any():
            bin_gt = gt_flat[bin_mask]
            bin_medians[i] = torch.median(bin_gt)
            print(f'[DEBUG] Bin {i} median: {bin_medians[i].item():.4f}')
        else:
            bin_medians[i] = (bins[i] + bins[i+1]) / 2.0
            print(f'[DEBUG] Bin {i} median (fallback): {bin_medians[i].item():.4f}')
    
    # Compute per-bin L1 loss
    total_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
    for i in range(len(bins)-1):
        bin_mask = bin_assignments == i
        if not bin_mask.any():
            print(f'[DEBUG] Bin {i} empty → skipping')
            continue
        
        bin_pred = pred_flat[bin_mask]
        bin_gt = gt_flat[bin_mask]
        
        bin_l1 = torch.abs(bin_pred - bin_gt).mean()
        print(f'[DEBUG] Bin {i} L1: {bin_l1.item():.6f}')
        
        weighted_loss = bin_l1 * bin_weights[i]
        print(f'[DEBUG] Bin {i} weighted loss: {weighted_loss.item():.6f}')
        
        if scale_by_median:
            weighted_loss = weighted_loss / (bin_medians[i] + eps)
            print(f'[DEBUG] Bin {i} scaled loss: {weighted_loss.item():.6f}')
        
        total_loss = total_loss + weighted_loss
        
    print(f'[DEBUG] Final loss: {total_loss.item():.6f}')
    return total_loss
