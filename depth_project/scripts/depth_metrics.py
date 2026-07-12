"""Standard monocular depth-estimation metrics.

All functions operate on torch tensors of shape (N, 1, H, W) or (N, H, W),
in metres.  Predictions are clipped to [0.5, 10.0] before evaluation.
"""
from __future__ import annotations

import torch

_MIN_DEPTH = 0.5
_MAX_DEPTH = 10.0


def _prep(pred, gt):
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)
    pred = pred.clamp(_MIN_DEPTH, _MAX_DEPTH).float()
    gt = gt.float()
    valid = gt > 0
    return pred, gt, valid


def compute_depth_metrics(pred, gt):
    pred_c, gt_c, valid = _prep(pred, gt)
    abs_rel = torch.zeros(1, device=pred.device)
    rmse_sum = torch.zeros(1, device=pred.device)
    mae_log = torch.zeros(1, device=pred.device)
    d1 = torch.zeros(1, device=pred.device)
    d2 = torch.zeros(1, device=pred.device)
    d3 = torch.zeros(1, device=pred.device)
    n_valid = torch.zeros(1, device=pred.device)
    for i in range(pred_c.shape[0]):
        p = pred_c[i][valid[i]]
        g = gt_c[i][valid[i]]
        if p.numel() == 0:
            continue
        abs_rel += (torch.abs(p - g) / g).sum()
        rmse_sum += ((p - g) ** 2).sum()
        mae_log += torch.abs(torch.log10(p) - torch.log10(g)).sum()
        ratio = torch.max(p / g, g / p)
        d1 += (ratio < 1.25).float().sum()
        d2 += (ratio < 1.25 ** 2).float().sum()
        d3 += (ratio < 1.25 ** 3).float().sum()
        n_valid += p.numel()
    n = n_valid.item()
    if n == 0:
        inf = float("inf")
        return {k: inf for k in ("AbsRel", "RMSE", "MAE", "delta1", "delta2", "delta3")}
    return {
        "AbsRel": abs_rel.item() / n,
        "RMSE": (rmse_sum.item() / n) ** 0.5,
        "MAE": mae_log.item() / n,
        "delta1": d1.item() / n,
        "delta2": d2.item() / n,
        "delta3": d3.item() / n,
    }


def format_metrics_line(metrics, prefix="EVAL"):
    parts = []
    for key in ("AbsRel", "RMSE", "MAE", "delta1", "delta2", "delta3"):
        if key in metrics:
            parts.append("val_%s=%.4f" % (key, metrics[key]))
    return prefix + " " + " ".join(parts)
