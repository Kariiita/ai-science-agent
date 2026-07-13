# V-SciAgent one-click demo script (PowerShell)
# Runs baseline training + agent loop end-to-end

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition

function Section($msg) { Write-Host "`n==== $msg ====" -ForegroundColor Cyan }

# 1. Check Python version
Section "Checking Python"
$pyVer = python --version 2>&1
Write-Host "Python: $pyVer"
if (-not ($pyVer -match "3\.\d+")) { Write-Host "ERROR: Python 3.x required" -ForegroundColor Red; exit 1 }

# 2. Check GPU
Section "Checking GPU"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: PyTorch not installed. Install with:" -ForegroundColor Yellow
    Write-Host "  pip install torch-2.11.0+cu128-cp312-cp312-win_amd64.whl torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl"
}

# 3. Install dependencies
Section "Installing dependencies"
pip install -r requirements.txt --quiet 2>&1 | Out-Null
Write-Host "Dependencies installed."

# 4. Check API keys
Section "Checking API keys"
if (-not $env:DASHSCOPE_API_KEY) {
    Write-Host "WARNING: DASHSCOPE_API_KEY not set. Agent will not work." -ForegroundColor Yellow
    Write-Host "  Set with: `$env:DASHSCOPE_API_KEY = 'sk-xxx'"
} else {
    Write-Host "DASHSCOPE_API_KEY: SET"
}
if (-not $env:GLM_CODING_PLAN_API_KEY) {
    Write-Host "WARNING: GLM_CODING_PLAN_API_KEY not set. Literature search will use local database only." -ForegroundColor Yellow
    Write-Host "  Set with: `$env:GLM_CODING_PLAN_API_KEY = 'xxx'"
} else {
    Write-Host "GLM_CODING_PLAN_API_KEY: SET"
}

# 5. Check dataset
Section "Checking dataset"
$dataDir = Join-Path $ROOT "depth_project\data\train\rgb"
if (Test-Path $dataDir) {
    $count = (Get-ChildItem $dataDir -Filter *.png).Count
    Write-Host "Training images: $count"
    if ($count -eq 0) {
        Write-Host "ERROR: No training images found. Prepare NYU Depth v2 data first." -ForegroundColor Red
        Write-Host "  Option A: Download nyu_depth_v2_labeled.mat and run scripts\convert_nyu_mat.py"
        Write-Host "  Option B: Place RGB+depth pairs in data\train\{rgb,depth}\ and data\val\{rgb,depth}\"
        exit 1
    }
} else {
    Write-Host "ERROR: data\train\rgb\ not found. Prepare dataset first." -ForegroundColor Red
    exit 1
}

# 6. Run baseline training (1 cycle)
Section "Running baseline training"
Push-Location (Join-Path $ROOT "depth_project")
python scripts\train_dorn.py --data_dir data --epochs 5 --batch_size 4
Pop-Location

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Training failed." -ForegroundColor Red
    exit 1
}

# 7. Run agent loop (3 cycles)
Section "Running agent loop (3 cycles)"
Push-Location $ROOT
python -c "from api import AutoResearcher; r = AutoResearcher('depth_project'); r.run_n_cycles(3)"
Pop-Location

# 8. Summary
Section "Demo complete"
Write-Host "Check results:"
Write-Host "  - depth_project\logs\           (training logs)"
Write-Host "  - depth_project\model_snapshots\ (saved models)"
Write-Host "  - depth_project\experiment_history.db (cycle history)"
Write-Host ""
Write-Host "To start HTTP API:"
Write-Host "  uvicorn web.app:app --host 0.0.0.0 --port 8000"
Write-Host "  Open http://localhost:8000/docs for Swagger UI"
