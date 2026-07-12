# Goal
Improve monocular depth estimation on NYU Depth v2. Target: val_AbsRel < 0.15, val_RMSE < 0.5.

# Codebase
- models/: PyTorch depth estimation networks (nn.Module subclasses)
- datasets/: Dataset loaders returning (RGB image, depth ground truth) pairs
- data/: training and validation data (images + depth maps)

# Constraints
- Single GPU (NVIDIA RTX 5060, 8GB)
- Max 50 epochs per experiment
- Use PyTorch
- Evaluate with AbsRel, RMSE, MAE, delta1 (threshold 1.25)

# Success Criteria
- val_AbsRel < 0.15 on validation set
- val_RMSE < 0.5
- delta1 > 0.75
