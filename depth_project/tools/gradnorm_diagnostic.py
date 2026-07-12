import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.nyu_depth_v2 import NYUDepthV2
from torch.utils.data import DataLoader
from scripts.train_dorn_gradnorm import _load_model

def compute_gradnorm_stats(model, val_loader, device, num_batches=10):
    """Compute GradNorm gradient norm stability on first `num_batches` of validation set."""
    model.eval()
    grad_norms = []
    
    with torch.no_grad():
        for i, (images, depths) in enumerate(val_loader):
            if i >= num_batches:
                break
            images, depths = images.to(device), depths.to(device)
            outputs = model(images)
            
            # Compute spatial gradients
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                                 dtype=torch.float32, device=outputs.device)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                                 dtype=torch.float32, device=outputs.device)
            
            dx = F.conv2d(outputs, sobel_x, padding=1)
            dy = F.conv2d(outputs, sobel_y, padding=1)
            
            # Gradient magnitude
            grad_mag = torch.sqrt(dx**2 + dy**2 + 1e-8)
            
            # Per-batch gradient norm
            batch_norm = grad_mag.norm().item()
            grad_norms.append(batch_norm)
            
            print(f'Batch {i+1}: grad_norm = {batch_norm:.4f}')
    
    grad_norms = torch.tensor(grad_norms)
    mean_norm = grad_norms.mean().item()
    std_norm = grad_norms.std().item()
    max_norm = grad_norms.max().item()
    
    print(f'\nGradNorm Statistics over {num_batches} batches:')
    print(f'Mean: {mean_norm:.4f}, Std: {std_norm:.4f}, Max: {max_norm:.4f}')
    
    # Clip if > 1.0
    if max_norm > 1.0:
        print('WARNING: Max gradient norm > 1.0 — consider clipping')
    else:
        print('OK: All gradient norms <= 1.0')
    
    return grad_norms

if __name__ == "__main__":
    # Load model and data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Use same config as training
    data_dir = 'data'
    batch_size = 8
    
    try:
        val_dataset = NYUDepthV2(root=data_dir, split='val')
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0)
        print(f'Loaded {len(val_dataset)} validation samples')
    except Exception as e:
        print(f'Error loading dataset: {e}')
        raise
    
    try:
        model = _load_model('scripts.dorn:dorn_baseline').to(device)
        print('Model loaded successfully')
    except Exception as e:
        print(f'Error loading model: {e}')
        raise
    
    # Compute stats
    grad_norms = compute_gradnorm_stats(model, val_loader, device)
    
    print('Diagnostic completed!')
    import os
    os._exit(0)
