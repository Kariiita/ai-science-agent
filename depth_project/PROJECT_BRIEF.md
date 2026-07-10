# Depth Estimation Project

## Goal
Train a CNN-based depth estimation model on NYU Depth v2 style data.
Target: val_AbsRel < 0.20

## Codebase
- scripts/: Training and evaluation scripts
- data/: Training and validation data (RGB + depth pairs)

## Constraints
- Single GPU (RTX 5060 Laptop, 8GB)
- Max 50 epochs per experiment
- Use PyTorch

## Success Criteria
- val_AbsRel < 0.20
- val_RMSE < 0.75
- val_MAE < 0.25