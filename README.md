# Digital Twin ICU — Mamba SSM for Clinical Forecasting

Selective State-Space Model (Mamba) for ICU patient physiological trajectory prediction and counterfactual simulation.

## Architecture

- **Model**: Full-sequence Mamba SSM (d_model=64, d_state=16, d_conv=4)
- **CDSP**: Orthogonal decorrelation penalty (treatment bias reduction)
- **Hawkes**: Latent conditional intensity regularization
- **Loss**: Masked MSE (dense vitals + α·sparse labs), only observed targets
- **Input**: 15 vitals/labs + 5 treatments → 50 dims (x + mask + delta_t + a)
- **Windows**: 72h context → 24h prediction (2h bins, stride=2h)

## Files

| File | Purpose |
|---|---|
| `train.py` | **Main training script** (portable, configurable) |
| `visualize_results.py` | Generate all plots from saved results |
| `digital_twin_v5.py` | Self-contained version (local testing) |
| `requirements.txt` | Python dependencies |

## Quick Start (DCC)

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/YOUR_USERNAME/digitaltwinICU.git
cd digitaltwinICU

# Create conda environment with CUDA support
conda create -n twin python=3.10 -y
conda activate twin

# Install PyTorch with CUDA (check DCC's CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install Mamba (requires CUDA)
pip install causal-conv1d>=1.0.0
pip install mamba-ssm>=1.0.0

# Install remaining deps
pip install -r requirements.txt
```

### 2. Place Data

```bash
# Create data directory
mkdir -p data/mimic_iv_parquet_files

# Copy data files (from your local machine or shared storage)
# Option A: SCP from local
scp -r /path/to/mimic_iv_parquet_files/*.parquet dcc:~/digitaltwinICU/data/mimic_iv_parquet_files/
scp /path/to/icustays2.csv dcc:~/digitaltwinICU/data/

# Option B: If data is on DCC shared storage, symlink
ln -s /work/shared/mimic_iv_parquet_files data/mimic_iv_parquet_files
ln -s /work/shared/icustays2.csv data/icustays2.csv
```

Expected data structure:
```
data/
├── mimic_iv_parquet_files/
│   ├── 0.parquet
│   ├── 1.parquet
│   └── ... (75 files, ~1.2GB total)
└── icustays2.csv (14MB)
```

### 3. Train (Full Dataset)

```bash
# Full dataset (~65K patients), 50 epochs
python train.py \
    --data_dir ./data \
    --output_dir ./results \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --num_workers 4

# Or test with 2% first:
python train.py --data_dir ./data --subjects 1307 --epochs 30
```

### 4. Visualize Results

```bash
python visualize_results.py --results_dir ./results
```

This generates:
- `overview.png` — 20-panel summary
- `per_feature_metrics.png` — per-feature MSE/MAE/RMSE/R² table
- `best_patient_all_features.png` — best prediction, all 15 features
- `loss_curves.png` — train/test/CDSP/Hawkes curves
- `rollout_example.png` — autoregressive rollout with uncertainty

### 5. Saved Files (in `results/`)

| File | Description |
|---|---|
| `best_model.pt` | Best model weights (by test loss) |
| `final_model.pt` | Final epoch model weights |
| `training_history.json` | All loss curves |
| `test_predictions.npz` | All test predictions + ground truth |
| `metrics.csv` | Per-window MSE/MAE/RMSE/R²/Pearson |
| `config.json` | Training configuration |
| `rollout_example.npz` | One rollout example for visualization |

## SLURM (if needed)

```bash
#!/bin/bash
#SBATCH --job-name=digital_twin
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.log

conda activate twin
python train.py --data_dir ./data --output_dir ./results --epochs 50 --batch_size 64 --num_workers 4
python visualize_results.py --results_dir ./results
```

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--hidden_dim` | 64 | Mamba hidden dimension |
| `--alpha` | 1.0 | Sparse lab loss weight |
| `--lambda_cdsp` | 0.1 | CDSP regularization weight |
| `--lambda_hawkes` | 0.01 | Hawkes regularization weight |
| `--cdsp_warmup` | 3 | Epochs before CDSP activates |
| `--subjects` | None | Max subjects (None=all) |
