import torch
import torch.nn as nn
import torch.nn.functional as F


def gradnorm_loss(pred, gt=None, K=None, weight=1.0, clip_grad_norm=1.0):
    """
    GradNorm loss: penalize deviation from planar gradient structure.
    
    Args:
        pred: predicted depth map, shape (B, 1, H, W)
        gt: optional ground truth depth for masking (B, 1, H, W)
        K: camera intrinsics matrix, shape (3, 3) or (B, 3, 3)
        weight: scalar weight for the loss term
        clip_grad_norm: max allowed gradient norm (clips if exceeded)
    
    Returns:
        loss: scalar tensor
    """
    if K is None:
        # Default NYU intrinsics
        K = torch.tensor([[518.8579, 0.0, 320.0],
                         [0.0, 518.8579, 240.0],
                         [0.0, 0.0, 1.0]], device=pred.device)
    
    B, C, H, W = pred.shape
    assert C == 1, f'Expected single-channel depth, got {C}'
    
    # Compute spatial gradients using Sobel filters
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                         dtype=torch.float32, device=pred.device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                         dtype=torch.float32, device=pred.device)
    
    # Apply Sobel filters
    dx = F.conv2d(pred, sobel_x, padding=1)
    dy = F.conv2d(pred, sobel_y, padding=1)
    
    # Gradient magnitude
    grad_mag = torch.sqrt(dx**2 + dy**2 + 1e-8)
    
    # Clip gradient magnitude if norm exceeds threshold
    grad_norm = grad_mag.norm()
    if grad_norm > clip_grad_norm:
        grad_mag = grad_mag * (clip_grad_norm / grad_norm)
    
    # Penalize spatial variation of gradient magnitude
    mean_grad_mag = grad_mag.mean(dim=[2,3], keepdim=True)
    grad_var = ((grad_mag - mean_grad_mag)**2).mean()
    
    # Also penalize large gradients (encourages smoothness)
    grad_smooth = grad_mag.mean()
    
    # Total GradNorm loss
    loss = grad_var + 0.1 * grad_smooth
    
    # Apply near-field depth masking: compute GradNorm only on pixels where gt < 2.0m
    if gt is not None:
        # Create mask for near-field pixels (gt < 2.0)
        mask = (gt < 2.0).float()
        # Apply mask to gradient magnitude
        masked_grad_mag = grad_mag * mask
        # Recompute mean and variance with mask
        masked_mean_grad_mag = (masked_grad_mag * mask).sum(dim=[2,3], keepdim=True) / (mask.sum(dim=[2,3], keepdim=True) + 1e-8)
        masked_grad_var = ((masked_grad_mag - masked_mean_grad_mag)**2 * mask).sum() / (mask.sum() + 1e-8)
        # Recompute smoothness with mask
        masked_grad_smooth = (masked_grad_mag * mask).sum() / (mask.sum() + 1e-8)
        # Update loss with masked components
        loss = masked_grad_var + 0.1 * masked_grad_smooth
    
    return loss * weight


def get_decoder_grad_norms(model, pred, gt, K=None, weight=1.0):
    """
    Compute per-layer gradient norms for decoder layers only.
    Returns dict mapping layer name to gradient norm.
    """
    if K is None:
        K = torch.tensor([[518.8579, 0.0, 320.0],
                         [0.0, 518.8579, 240.0],
                         [0.0, 0.0, 1.0]], device=pred.device)
    
    # Compute GradNorm loss
    loss = gradnorm_loss(pred, gt=gt, K=K, weight=weight)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass
    loss.backward(retain_graph=True)
    
    # Collect norms for decoder layers
    norms = {}
    for name, param in model.named_parameters():
        # Identify decoder layers: upconv and final_conv
        if 'upconv' in name or 'final_conv' in name:
            if param.grad is not None:
                norms[name] = param.grad.norm().item()
    
    return norms
