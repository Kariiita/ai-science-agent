import torch
import torch.nn as nn
import torch.nn.functional as F


def depth_binned_l1_loss(pred, gt, bins=None, bin_weights=None, scale_by_median=True, eps=1e-6):
    """
    Depth-binned, scale-aware L1 loss.
    
    Divides depth range [0,10] into 5 bins (0–2, 2–4, 4–6, 6–8, 8–10m);
    applies per-bin L1 loss weighted by bin inverse frequency
    (to upweight underrepresented near-field regions)
    and scale-normalized by bin median depth.
    
    Args:
        pred: predicted depth map, shape (B, 1, H, W)
        gt: ground truth depth map, shape (B, 1, H, W)
        bins: list of bin edges, e.g. [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        bin_weights: optional list of per-bin weights (length = len(bins)-1)
        scale_by_median: if True, normalize each bin's loss by bin median depth
        eps: small constant for numerical stability
    
    Returns:
        loss: scalar tensor
    """
    if bins is None:
        bins = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    
    # Flatten predictions and ground truth
    pred_flat = pred.view(-1)
    gt_flat = gt.view(-1)
    
    # Mask valid pixels (gt > 0)
    valid_mask = gt_flat > 0
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]
    
    if pred_flat.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # Assign each pixel to a bin
    bin_assignments = torch.zeros_like(gt_flat, dtype=torch.long)
    for i in range(len(bins) - 1):
        mask = (gt_flat >= bins[i]) & (gt_flat < bins[i+1])
        bin_assignments[mask] = i
    
    # Compute bin frequencies
    bin_counts = torch.zeros(len(bins)-1, device=pred.device)
    for i in range(len(bins)-1):
        bin_counts[i] = (bin_assignments == i).sum().float()
    
    # Compute bin weights: inverse frequency normalized to sum to 1
    if bin_weights is None:
        # Avoid division by zero: replace zeros with eps before inversion
        bin_counts_safe = bin_counts + eps
        bin_weights = 1.0 / bin_counts_safe
        bin_weights = bin_weights / bin_weights.sum()
    
    # Compute bin medians for scaling
    bin_medians = torch.zeros(len(bins)-1, device=pred.device)
    for i in range(len(bins)-1):
        bin_mask = bin_assignments == i
        if bin_mask.any():
            bin_gt = gt_flat[bin_mask]
            bin_medians[i] = torch.median(bin_gt)
        else:
            bin_medians[i] = (bins[i] + bins[i+1]) / 2.0
    
    # Compute per-bin L1 loss
    total_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
    for i in range(len(bins)-1):
        bin_mask = bin_assignments == i
        if not bin_mask.any():
            continue
        
        bin_pred = pred_flat[bin_mask]
        bin_gt = gt_flat[bin_mask]
        
        # Per-bin L1 loss
        bin_l1 = torch.abs(bin_pred - bin_gt).mean()
        
        # Apply weight
        weighted_loss = bin_l1 * bin_weights[i]
        
        # Scale by median depth if requested
        if scale_by_median:
            weighted_loss = weighted_loss / (bin_medians[i] + eps)
        
        total_loss = total_loss + weighted_loss
    
    return total_loss
