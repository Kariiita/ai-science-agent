'''Train DORN baseline on NYU Depth v2 with GradNorm loss, depth-aware gradient clipping, and depth-binned L1 loss.

Usage:
    python scripts/train_dorn_gradnorm_depthclip_binned_l1.py --data_dir data --epochs 50 --batch_size 4
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
from tools.depth_binned_l1_loss import depth_binned_l1_loss


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
    parser.add_argument("--batch_size", type=int, default=8)
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
    eval_every = max(1, args.epochs // 10)

    # Camera intrinsics for NYU Depth v2
    K = torch.tensor([[518.8579, 0.0, 320.0],
                     [0.0, 518.8579, 240.0],
                     [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for i, (images, depths) in enumerate(train_loader):
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            # Compute depth-binned L1 loss
            binned_l1_loss = depth_binned_l1_loss(outputs, depths)
            
            # Compute GradNorm gradient consistency loss
            grad_loss = gradnorm_loss(outputs, gt=depths, K=K, weight=0.1)
            
            # Combine losses with GradNorm alpha=0.7
            # Total loss = 0.7 * binned_l1_loss + 0.3 * grad_loss
            total_loss = 0.7 * binned_l1_loss + 0.3 * grad_loss
            
            total_loss.backward()
            
            # Depth-aware gradient clipping for last 3 decoder layers
            # Clip gradients for upconv2, upconv3, and final_conv
            for name, param in model.named_parameters():
                if 'upconv2.weight' in name or 'upconv3.weight' in name or 'final_conv.weight' in name:
                    if param.grad is not None:
                        # Clip based on depth magnitude: deeper regions get smaller clip norm
                        # Use median depth of current batch as scaling factor
                        median_depth = torch.median(depths[depths > 0])
                        clip_norm = 1.0 / (median_depth + 1e-6)
                        torch.nn.utils.clip_grad_norm_(param, clip_norm)
            
            optimizer.step()
            train_loss += total_loss.item()
            n_batches += 1
            
            if i % 20 == 0:
                log(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Step [{i}/{len(train_loader)}] binned_l1={binned_l1_loss.item():.4f}, "
                    f"grad={grad_loss.item():.4f}, total={total_loss.item():.4f}")

        avg_loss = train_loss / max(n_batches, 1)
        log(f"Epoch [{epoch+1}/{args.epochs}] avg_train_loss={avg_loss:.4f}")

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
                           os.path.join("model_snapshots", "best_model_gradnorm_depthclip_binned_l1.pth"))
                log(f"Saved best_model_gradnorm_depthclip_binned_l1.pth (val_AbsRel={best_abs_rel:.4f})")

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
    import os as _os
    _os._exit(0)  # Force exit to avoid CUDA thread hang on Windows


if __name__ == "__main__":
    main()
