"""Train DORN baseline on NYU Depth v2 with full depth metrics.

Usage:
    python scripts/train_dorn.py --data_dir data --epochs 5 --batch_size 4
"""
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


def _load_model(spec: str):
    """Load model from 'module:function' spec, e.g. 'scripts.dorn:dorn_baseline'."""
    if ":" in spec:
        module_name, func_name = spec.split(":", 1)
    else:
        module_name, func_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    return factory()


def _silog_loss(pred, gt):
    """Scale-invariant log loss (Eigen et al.)."""
    mask = gt > 0
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)
    log_diff = torch.log(pred[mask]) - torch.log(gt[mask])
    n = log_diff.numel()
    return (log_diff ** 2).mean() - 0.5 * (log_diff.sum() ** 2) / (n * n)


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
    criterion = nn.L1Loss()
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

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for i, (images, depths) in enumerate(train_loader):
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
            if i % 20 == 0:
                log(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Step [{i}/{len(train_loader)}] loss={loss.item():.4f}")

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
                           os.path.join("model_snapshots", "best_model.pth"))
                log(f"Saved best_model.pth (val_AbsRel={best_abs_rel:.4f})")

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
    log("Training completed!
    os._exit(0)")
    log_fh.close()


if __name__ == "__main__":
    main()
