import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian_pyramid_residuals(x, num_levels=4):
    """
    Compute Laplacian pyramid residuals at multiple scales.
    Returns list of residuals: [full_res, half_res, quarter_res, eighth_res]
    """
    residuals = []
    current = x
    
    for level in range(num_levels):
        # Downsample by factor 2
        if level > 0:
            down_h = current.shape[2] // 2
            down_w = current.shape[3] // 2
            downsampled = F.interpolate(current, size=(down_h, down_w), mode='bilinear', align_corners=False)
            
            # Upsample back to original size
            upsampled = F.interpolate(downsampled, size=(current.shape[2], current.shape[3]), mode='bilinear', align_corners=False)
            
            # Residual = original - upsampled
            residual = current - upsampled
            residuals.append(residual)
            
            # Current becomes downsampled for next level
            current = downsampled
        else:
            # Full resolution residual is just the input
            residuals.append(current)
    
    return residuals


def near_field_laplacian_loss(pred, gt, near_range=(0.0, 5.0), eps=1e-6):
    """
    Near-field (0–5m) Laplacian pyramid loss with variance suppression.
    
    Computes Laplacian residuals only within near_range mask,
    then applies L2 penalty scaled by inverse depth variance in that bin.
    
    Args:
        pred: predicted depth map, shape (B, 1, H, W)
        gt: ground truth depth map, shape (B, 1, H, W)
        near_range: tuple (min_depth, max_depth) for near-field mask
        eps: small constant for numerical stability
    
    Returns:
        loss: scalar tensor
    """
    min_d, max_d = near_range
    
    # Valid mask: gt > 0 AND within near range
    valid_mask = (gt > 0) & (gt >= min_d) & (gt < max_d)
    
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # Compute Laplacian pyramid residuals
    residuals = laplacian_pyramid_residuals(pred, num_levels=4)
    
    # Compute depth variance *within near-field mask*
    gt_near = gt[valid_mask]
    if gt_near.numel() == 0:
        var_near = torch.tensor(1.0, device=pred.device)
    else:
        var_near = torch.var(gt_near, unbiased=False) + eps
    
    # Inverse variance scaling factor
    scale_factor = 1.0 / var_near
    
    total_loss = 0.0
    
    # Apply L2 loss on residuals *only where near_mask is True*
    for scale_idx, residual in enumerate(residuals):
        # Resize near_mask to match residual spatial dims
        mask_resized = F.interpolate(
            valid_mask.float(), 
            size=(residual.shape[2], residual.shape[3]), 
            mode='nearest'
        )
        
        # Masked residual
        masked_residual = residual * mask_resized
        
        # L2 loss: mean of squared values over masked region
        # Use sum / count to avoid division by zero
        sq_sum = (masked_residual ** 2).sum()
        num_valid = mask_resized.sum()
        
        if num_valid > 0:
            l2_loss = sq_sum / num_valid
            # Scale by inverse variance and level weight (full-res most important)
            level_weight = 1.0 / (2 ** scale_idx)
            total_loss += l2_loss * level_weight
    
    return total_loss * scale_factor
