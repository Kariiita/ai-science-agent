"""Reflection Agent validation - generates a sample reflection record."""
import os, json
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(r"D:\code\ai-science-agent-team\depth_project\workspace\reflections")
WORKSPACE.mkdir(parents=True, exist_ok=True)

cycle = 1
record = f"""## Cycle {cycle} Reflection

### Hypothesis
If we add a depth-aware loss (SilogLoss) to the baseline CNN, then val_AbsRel should
improve by at least 5% because SilogLoss handles the multi-scale depth distribution
better than plain L1 loss.

### Changes Made
- Replaced L1Loss with SilogLoss in train_baseline.py
- Added invalid depth pixel masking (zero-depth regions excluded from loss)
- Learning rate: 0.001, batch size: 8, epochs: 10

### Metrics
| Metric | Before | After | Target | Pass? |
|--------|--------|-------|--------|-------|
| AbsRel | 0.250  | 0.220 | < 0.20 | No    |
| RMSE   | 0.850  | 0.780 | < 0.75 | No    |
| MAE    | 0.300  | 0.270 | < 0.25 | No    |
| d1     | 0.650  | 0.700 | > 0.75 | No    |

### Reflection
The SilogLoss improved all metrics by ~10%, confirming that scale-invariant losses
help with the multi-scale depth distribution. However, we did not meet the target
thresholds. The improvement suggests the direction is correct but the model capacity
or training duration may be insufficient. The Data Agent report showed 5% invalid
depth pixels which we successfully masked.

### Next Iteration Plan
1. Increase model capacity: add more decoder layers (UNet style)
2. Train for 50 epochs instead of 10
3. Add data augmentation: random horizontal flip + color jitter
4. Consider learning rate scheduling (cosine annealing)

### Causal Link
Switching from L1Loss to SilogLoss caused val_AbsRel to improve from 0.250 to 0.220
because SilogLoss penalizes relative errors equally across depth ranges, addressing
the scale imbalance in the depth distribution.
"""

reflect_path = WORKSPACE / f"cycle_{cycle}.md"
with open(reflect_path, "w", encoding="utf-8") as f:
    f.write(record)

summary = {
    "cycle": cycle,
    "decision": "Increase model capacity with UNet decoder, train 50 epochs, add augmentation",
    "milestone": "SilogLoss improved AbsRel by 12% (0.250 -> 0.220), direction confirmed but target not met",
    "dead_end": None,
    "active_problem": "val_AbsRel remains at 0.220, target is < 0.20",
    "causal_link": "SilogLoss caused val_AbsRel to improve 0.250->0.220 because scale-invariant loss handles multi-scale depth better",
    "lesson": "Mask invalid depth pixels before computing loss to avoid gradient pollution"
}

print("=== Reflection Record (cycle_1.md) ===")
print(record)
print("\n=== Reflection Summary (JSON) ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))