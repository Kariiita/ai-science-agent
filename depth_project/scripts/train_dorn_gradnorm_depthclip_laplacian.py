'''Train DORN baseline on NYU Depth v2 with GradNorm loss, depth-aware gradient clipping, and depth-aware Laplacian pyramid loss.

Usage:
    python scripts/train_dorn_gradnorm_depthclip_laplacian.py --data_dir data --epochs 50 --batch_size 4
'''
import argparse
import importlib
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.nyu_depth_v2 import NYUDepthV2
from scripts.depth_metrics import compute_depth_metrics, format_metrics_line
from tools.gradnorm_loss import gradnorm_loss
from tools.laplacian_depth_loss import depth_aware_laplacian_loss


def _load_model(spec: str):
    """Load model from 'module:function' spec, e.g. 'scripts.dorn:dorn_baseline'."""
    if ':' in spec:
        module_name, func_name = spec.split(':', 1)
    else:
        module_name, func_name = spec.rsplit('.', 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    return factory()


def main():
    parser = argparse.ArgumentParser(description="Train DORN baseline on NYU Depth v2")
    parser.add_argument("--model", type=str, default="scripts.dorn:dorn_baseline",
                        help="Model factory as module:function")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config file (CLI args take priority)")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f).get("train", {})
        for key in ("batch_size", "epochs", "lr"):
            if key in cfg:
                setattr(args, key, cfg[key])

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_dataset = NYUDepthV2(root=args.data_dir, split="train")
    val_dataset = NYUDepthV2(root=args.data_dir, split="val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    model = _load_model(args.model).to(device)
    l1_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("model_snapshots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"train_{int(time.time())}.log")
    log_fh = open(log_path, "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    log(f"Train config: model={args.model}, data={args.data_dir}, "
        f"batch={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    log(f"Device: {device}, CUDA: {torch.cuda.is_available()}")
    log(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    best_abs_rel = float("inf")
    best_epoch = -1
    eval_every = 5  # Validate every 5 epochs

    # Camera intrinsics for NYU Depth v2
    K = torch.tensor([[518.8579, 0.0, 320.0],
                     [0.0, 518.8579, 240.0],
                     [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        max_grad_norm = 0.0
        cuda_memory_max = 0.0
        for i, (images, depths) in enumerate(train_loader):
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            # Compute L1 reconstruction loss
            l1_loss = l1_criterion(outputs, depths)
            
            # Compute GradNorm gradient consistency loss
            grad_loss = gradnorm_loss(outputs, gt=depths, K=K, weight=0.1)
            
            # Compute depth-aware Laplacian pyramid loss
            laplacian_loss = depth_aware_laplacian_loss(outputs, depths, bins=[(0.0, 5.0), (5.0, 10.0)], weight=0.5)
            
            # Combine losses
            total_loss = l1_loss + grad_loss + laplacian_loss
            
            total_loss.backward()
            
            # Clip gradients for last 3 decoder layers (upconv2, upconv3, final_conv)
            # We'll clip upconv2, upconv3, and final_conv (3 layers)
            layers_to_clip = ['upconv2.weight', 'upconv3.weight', 'final_conv.weight']
            for name, param in model.named_parameters():
                if name in layers_to_clip and param.grad is not None:
                    # Depth-aware gradient clipping: clip based on depth bin
                    # For pixels in 0-5m bin, use smaller clip norm; for 5-10m, use larger
                    # First, get the depth values for this batch
                    depth_bin_mask_0_5 = (depths >= 0.0) & (depths < 5.0)
                    depth_bin_mask_5_10 = (depths >= 5.0) & (depths < 10.0)
                    
                    # Apply different clipping norms per bin
                    if depth_bin_mask_0_5.any():
                        # Clip gradients for 0-5m bin with norm 0.5
                        torch.nn.utils.clip_grad_norm_(param, max_norm=0.5)
                    if depth_bin_mask_5_10.any():
                        # Clip gradients for 5-10m bin with norm 1.0
                        torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
            
            # Log max gradient norm
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm = max(grad_norm, p.grad.norm().item())
            max_grad_norm = max(max_grad_norm, grad_norm)
            
            # Log CUDA memory usage
            if torch.cuda.is_available():
                cuda_memory = torch.cuda.memory_allocated(device) / 1024**2
                cuda_memory_max = max(cuda_memory_max, cuda_memory)
            
            optimizer.step()
            train_loss += total_loss.item()
            n_batches += 1
            
            if i % 20 == 0:
                log(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Step [{i}/{len(train_loader)}] l1={l1_loss.item():.4f}, "
                    f"grad={grad_loss.item():.4f}, laplacian={laplacian_loss.item():.4f}, total={total_loss.item():.4f}")

        avg_loss = train_loss / max(n_batches, 1)
        log(f"Epoch [{epoch+1}/{args.epochs}] avg_train_loss={avg_loss:.4f}, "
            f"max_grad_norm={max_grad_norm:.4f}, cuda_mem_mb={cuda_memory_max:.1f}")

        # Evaluate periodically
        if (epoch + 1) % eval_every == 0 or (epoch + 1) == args.epochs:
            model.eval()
            all_preds = []
            all_gts = []
            with torch.no_grad():
                for images, depths in val_loader:
                    images, depths = images.to(device), depths.to(device)
                    outputs = model(images)
                    all_preds.append(outputs.cpu())
                    all_gts.append(depths.cpu())
            preds = torch.cat(all_preds)
            gts = torch.cat(all_gts)
            m = compute_depth_metrics(preds, gts)
            log(format_metrics_line(m, prefix=f"EVAL epoch={epoch+1}"))
            if m["AbsRel"] < best_abs_rel:
                best_abs_rel = m["AbsRel"]
                best_epoch = epoch + 1
                torch.save(model.state_dict(),
                           os.path.join("model_snapshots", "best_model_gradnorm_depthclip_laplacian.pth"))
                log(f"Saved best_model_gradnorm_depthclip_laplacian.pth (val_AbsRel={best_abs_rel:.4f})")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join("model_snapshots", f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_abs_rel': best_abs_rel,
                'best_epoch': best_epoch
            }, checkpoint_path)
            log(f"Saved checkpoint: {checkpoint_path}")

    # Final summary
    model.eval()
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for images, depths in val_loader:
            images, depths = images.to(device), depths.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu())
            all_gts.append(depths.cpu())
    preds = torch.cat(all_preds)
    gts = torch.cat(all_gts)
    final = compute_depth_metrics(preds, gts)
    log(format_metrics_line(final, prefix="FINAL METRICS:"))
    log(f"Best val_AbsRel={best_abs_rel:.4f} at epoch {best_epoch}")
    log("Training completed!")
    log_fh.close()
    os._exit(0)


if __name__ == "__main__":
    main()
