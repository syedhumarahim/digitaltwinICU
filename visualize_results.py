"""
Digital Twin v5 — Visualization Script
=======================================
Loads saved results from train.py and generates all plots.

Usage:
    python visualize_results.py --results_dir ./results
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Feature config (must match train.py)
FEATURE_NAMES = [
    'Heart Rate', 'O2 Sat', 'NBP Mean', 'Resp Rate', 'Temp',
    'Creatinine', 'Lactate', 'MAP', 'Potassium', 'Sodium',
    'Chloride', 'BUN', 'Glucose', 'Platelets', 'Hemoglobin',
]
TREATMENT_NAMES = ['Med Start', 'Med Stop', 'IV NaCl', 'Turn', 'Intervention']
INPUT_DIM = 15
TREAT_DIM = 5
DENSE_INDICES = [0, 1, 2, 3]
SPARSE_INDICES = [4,5,6,7,8,9,10,11,12,13,14]
CONTEXT_STEPS = 36
HORIZON_STEPS = 12


def load_results(results_dir):
    """Load all saved results."""
    print(f"Loading results from {results_dir}/...")

    # Training history
    with open(os.path.join(results_dir, 'training_history.json')) as f:
        history = json.load(f)

    # Metrics
    metrics = pd.read_csv(os.path.join(results_dir, 'metrics.csv'))

    # Predictions
    data = np.load(os.path.join(results_dir, 'test_predictions.npz'))
    predictions = data['predictions']
    ground_truth = data['ground_truth']
    target_masks = data['target_masks']
    variances = data['variances']

    # Norm params
    with open(os.path.join(results_dir, 'norm_params.json')) as f:
        norm_params = json.load(f)

    # Rollout example
    rollout = None
    rollout_path = os.path.join(results_dir, 'rollout_example.npz')
    if os.path.exists(rollout_path):
        rollout = np.load(rollout_path)

    # Config
    config = None
    config_path = os.path.join(results_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    print(f"  ✓ {len(metrics)} test windows")
    print(f"  ✓ {len(history['train'])} training epochs")
    print(f"  ✓ Predictions shape: {predictions.shape}")
    return history, metrics, predictions, ground_truth, target_masks, variances, norm_params, rollout, config


# ============================================================
# Plot 1: 20-Panel Overview
# ============================================================

def plot_overview(history, metrics, predictions, ground_truth, target_masks, config, out_dir):
    """20-panel comprehensive visualization."""
    mdf = metrics

    fig = plt.figure(figsize=(28, 20))
    fig.suptitle('Digital Twin v5 — Full-Sequence SSM + CDSP + Hawkes',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. Learning curves
    ax = plt.subplot(4, 5, 1)
    ax.plot(history['train'], label='Train', lw=2.5, color='#2E86AB')
    ax.plot(history['test'], label='Test', lw=2.5, ls='--', color='#A23B72')
    ax.set_title('Learning Curves', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Masked Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Vital vs Lab loss
    ax = plt.subplot(4, 5, 2)
    ax.plot(history['vital'], label='Vitals', lw=2, color='#06A77D')
    ax.plot(history['lab'], label='Labs', lw=2, color='#F18F01')
    ax.set_title('Dense vs Sparse Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

    # 3. CDSP + Hawkes
    ax = plt.subplot(4, 5, 3)
    ax.plot(history['cdsp'], label='CDSP', lw=2, color='#E63946')
    ax.plot(history['hawkes'], label='Hawkes', lw=2, color='#457B9D')
    ax.set_title('Regularization', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

    # 4. MSE Distribution
    ax = plt.subplot(4, 5, 4)
    ax.hist(mdf['mse'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax.axvline(mdf['mse'].median(), color='red', ls='--',
               label=f"Med: {mdf['mse'].median():.4f}")
    ax.set_title('MSE Distribution', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 5. R² Distribution
    ax = plt.subplot(4, 5, 5)
    r2_clip = mdf['r2'].clip(-2, 1)
    ax.hist(r2_clip, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
    ax.axvline(r2_clip.median(), color='red', ls='--',
               label=f"Med: {r2_clip.median():.3f}")
    ax.set_title('R² Distribution', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 6-10: Best 5 (Heart Rate)
    best5 = mdf.nsmallest(5, 'mse').index.tolist()
    for i, idx in enumerate(best5):
        ax = plt.subplot(4, 5, 6 + i)
        time = np.arange(HORIZON_STEPS)
        tm = target_masks[idx][:, 0]
        obs_t = time[tm > 0.5]
        if len(obs_t) > 0:
            ax.plot(obs_t, ground_truth[idx][tm > 0.5, 0], 'o-', label='True',
                    ms=4, lw=1.5, color='#2E86AB')
        ax.plot(time, predictions[idx][:, 0], '--', label='Pred', lw=1.5, color='#F18F01')
        ax.set_title(f'Best #{i+1} HR (MSE={mdf.iloc[idx]["mse"]:.4f})', fontsize=10,
                    fontweight='bold', color='green')
        if i == 0: ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 11-15: Worst 5
    worst5 = mdf.nlargest(5, 'mse').index.tolist()
    for i, idx in enumerate(worst5):
        ax = plt.subplot(4, 5, 11 + i)
        time = np.arange(HORIZON_STEPS)
        tm = target_masks[idx][:, 0]
        obs_t = time[tm > 0.5]
        if len(obs_t) > 0:
            ax.plot(obs_t, ground_truth[idx][tm > 0.5, 0], 'o-',
                    ms=4, lw=1.5, color='#2E86AB')
        ax.plot(time, predictions[idx][:, 0], '--', lw=1.5, color='#A23B72')
        ax.set_title(f'Worst #{i+1} HR (MSE={mdf.iloc[idx]["mse"]:.4f})', fontsize=10,
                    fontweight='bold', color='red')
        ax.grid(True, alpha=0.3)

    # 16-18: Per-feature bar charts
    colors = ['#2E86AB' if f in DENSE_INDICES else '#F18F01' for f in range(INPUT_DIM)]
    for panel, met, title in [(16, 'mse', 'Per-Feature MSE'),
                               (17, 'r2', 'Per-Feature R²'),
                               (18, 'mae', 'Per-Feature MAE')]:
        ax = plt.subplot(4, 5, panel)
        vals = [mdf[f'{met}_f{f}'].mean() for f in range(INPUT_DIM)]
        ax.barh(range(INPUT_DIM), vals, color=colors, alpha=0.8)
        ax.set_yticks(range(INPUT_DIM))
        ax.set_yticklabels(FEATURE_NAMES, fontsize=7); ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    # 19. Pearson
    ax = plt.subplot(4, 5, 19)
    ax.hist(mdf['pearson'].dropna(), bins=20, color='#06A77D', alpha=0.7, edgecolor='black')
    ax.axvline(mdf['pearson'].mean(), color='red', ls='--',
               label=f"Mean: {mdf['pearson'].mean():.3f}")
    ax.set_title('Pearson Correlation', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 20. Summary table
    ax = plt.subplot(4, 5, 20)
    ax.axis('off')
    tbl = [['Metric', 'Value'],
           ['MSE', f"{mdf['mse'].mean():.4f}"],
           ['MAE', f"{mdf['mae'].mean():.4f}"],
           ['RMSE', f"{mdf['rmse'].mean():.4f}"],
           ['R²', f"{mdf['r2'].mean():.4f}"],
           ['Pearson', f"{mdf['pearson'].mean():.3f}"],
           ['Windows', f"{len(mdf)}"],
           ['Epochs', f"{len(history['train'])}"],
           ['Final Train', f"{history['train'][-1]:.4f}"],
           ['Final Test', f"{history['test'][-1]:.4f}"]]
    t = ax.table(cellText=tbl, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.8)
    for j in range(2):
        t[(0, j)].set_facecolor('#06A77D')
        t[(0, j)].set_text_props(weight='bold', color='white')

    plt.tight_layout()
    path = os.path.join(out_dir, 'overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================
# Plot 2: Per-Feature Table + Charts
# ============================================================

def plot_per_feature(metrics, out_dir):
    """Detailed per-feature MSE/MAE/RMSE/R² table and charts."""
    mdf = metrics

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle('Per-Feature Metrics (Masked — Observed Targets Only)',
                 fontsize=16, fontweight='bold')

    # Table
    axes[0].axis('off')
    header = ['Feature', 'Type', 'MSE', 'MAE', 'RMSE', 'R²']
    rows = []
    for f in range(INPUT_DIM):
        ft = 'Dense' if f in DENSE_INDICES else 'Sparse'
        m = mdf[f'mse_f{f}'].mean()
        a = mdf[f'mae_f{f}'].mean()
        r = mdf[f'r2_f{f}'].mean()
        rows.append([FEATURE_NAMES[f], ft, f'{m:.4f}', f'{a:.4f}',
                     f'{np.sqrt(abs(m)):.4f}', f'{r:.4f}'])
    cell = [header] + rows
    t = axes[0].table(cellText=cell, cellLoc='center', loc='center',
                      colWidths=[0.2, 0.1, 0.13, 0.13, 0.13, 0.13])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 2.0)
    for j in range(6):
        t[(0, j)].set_facecolor('#2E86AB')
        t[(0, j)].set_text_props(weight='bold', color='white')
    for i in range(1, 16):
        c = '#E8F4F8' if (i-1) in DENSE_INDICES else '#FFF3E0'
        for j in range(6): t[(i, j)].set_facecolor(c)

    # R² chart
    colors = ['#2E86AB' if f in DENSE_INDICES else '#F18F01' for f in range(INPUT_DIM)]
    feat_r2 = [mdf[f'r2_f{f}'].mean() for f in range(INPUT_DIM)]
    axes[1].barh(range(INPUT_DIM), feat_r2, color=colors, alpha=0.85)
    axes[1].set_yticks(range(INPUT_DIM))
    axes[1].set_yticklabels(FEATURE_NAMES, fontsize=10); axes[1].invert_yaxis()
    axes[1].set_xlabel('R²'); axes[1].axvline(0, color='black', lw=0.5)
    axes[1].set_title('R² per Feature', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].legend(handles=[Patch(color='#2E86AB', label='Dense'),
                            Patch(color='#F18F01', label='Sparse')], fontsize=10)

    # RMSE chart
    feat_rmse = [np.sqrt(abs(mdf[f'mse_f{f}'].mean())) for f in range(INPUT_DIM)]
    axes[2].barh(range(INPUT_DIM), feat_rmse, color=colors, alpha=0.85)
    axes[2].set_yticks(range(INPUT_DIM))
    axes[2].set_yticklabels(FEATURE_NAMES, fontsize=10); axes[2].invert_yaxis()
    axes[2].set_xlabel('RMSE (normalized)')
    axes[2].set_title('RMSE per Feature', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    path = os.path.join(out_dir, 'per_feature_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================
# Plot 3: Best Patient — All 15 Features
# ============================================================

def plot_best_patient(metrics, predictions, ground_truth, target_masks,
                      variances, norm_params, out_dir):
    """Best patient: all 15 features, actual vs predicted, with uncertainty."""
    best_idx = metrics['mse'].idxmin()
    pred = predictions[best_idx]
    true = ground_truth[best_idx]
    tm = target_masks[best_idx]
    var = variances[best_idx]
    np_params = norm_params[best_idx]

    fig, axes = plt.subplots(3, 5, figsize=(30, 15))
    fig.suptitle(f'Best Patient — All 15 Features (MSE={metrics.iloc[best_idx]["mse"]:.4f})',
                 fontsize=18, fontweight='bold')

    time_h = np.arange(HORIZON_STEPS) * 2  # Convert to hours

    for f in range(INPUT_DIM):
        ax = axes[f // 5, f % 5]
        tmf = tm[:, f] > 0.5
        obs_t = time_h[tmf]

        # De-normalize if we have params
        if np_params and np_params.get('mean'):
            mean_f = np_params['mean'][f]
            std_f = np_params['std'][f]
            true_raw = true[:, f] * std_f + mean_f
            pred_raw = pred[:, f] * std_f + mean_f
            std_raw = np.sqrt(var[:, f]) * std_f
        else:
            true_raw = true[:, f]
            pred_raw = pred[:, f]
            std_raw = np.sqrt(var[:, f])

        # True (observed only)
        if len(obs_t) > 0:
            ax.plot(obs_t, true_raw[tmf], 'o-', label='True',
                    ms=5, lw=2, color='#2E86AB')

        # Predicted (full line + uncertainty)
        ax.plot(time_h, pred_raw, '--', label='Predicted', lw=2, color='#F18F01')
        ax.fill_between(time_h, pred_raw - 2*std_raw, pred_raw + 2*std_raw,
                        alpha=0.15, color='#F18F01')

        ftype = 'Dense' if f in DENSE_INDICES else 'Sparse'
        ax.set_title(f'{FEATURE_NAMES[f]} [{ftype}]', fontsize=11, fontweight='bold')
        ax.set_xlabel('Hours')
        ax.grid(True, alpha=0.3)
        if f == 0: ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, 'best_patient_all_features.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================
# Plot 4: Training Loss Curves (Detailed)
# ============================================================

def plot_loss_curves(history, out_dir):
    """Detailed 4-panel training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Loss Curves', fontsize=16, fontweight='bold')

    # Train vs Test
    axes[0, 0].plot(history['train'], label='Train', lw=2.5, color='#2E86AB')
    axes[0, 0].plot(history['test'], label='Test', lw=2.5, ls='--', color='#A23B72')
    axes[0, 0].set_title('Reconstruction Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Masked MSE')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # Vital vs Lab
    axes[0, 1].plot(history['vital'], label='Vitals (Dense)', lw=2, color='#06A77D')
    axes[0, 1].plot(history['lab'], label='Labs (Sparse)', lw=2, color='#F18F01')
    axes[0, 1].set_title('Dense vs Sparse Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # CDSP
    axes[1, 0].plot(history['cdsp'], lw=2, color='#E63946')
    axes[1, 0].set_title('CDSP Decorrelation Loss', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].grid(True, alpha=0.3)

    # Hawkes
    axes[1, 1].plot(history['hawkes'], lw=2, color='#457B9D')
    axes[1, 1].set_title('Hawkes Intensity Loss', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'loss_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================
# Plot 5: Rollout Example
# ============================================================

def plot_rollout(rollout, norm_params, out_dir):
    """Show rollout: context → autoregressive prediction vs actual."""
    if rollout is None:
        print("  ⚠ No rollout example saved, skipping")
        return

    pred = rollout['prediction'][0]       # [12, 15]
    var = rollout['variance'][0]           # [12, 15]
    context = rollout['context']           # [36, 15]
    target = rollout['target']             # [12, 15]
    tm = rollout['target_mask']            # [12, 15]

    fig, axes = plt.subplots(3, 5, figsize=(30, 15))
    fig.suptitle('Autoregressive Rollout — Full Context Warmup → Prediction',
                 fontsize=18, fontweight='bold')

    ctx_time = np.arange(CONTEXT_STEPS) * 2
    tgt_time = np.arange(CONTEXT_STEPS, CONTEXT_STEPS + HORIZON_STEPS) * 2

    for f in range(INPUT_DIM):
        ax = axes[f // 5, f % 5]

        # Context
        ax.plot(ctx_time, context[:, f], lw=1, color='#999999', alpha=0.6, label='Context')

        # True target (observed only)
        tmf = tm[:, f] > 0.5
        obs_t = tgt_time[tmf]
        if len(obs_t) > 0:
            ax.plot(obs_t, target[tmf, f], 'o-', ms=5, lw=2,
                    color='#2E86AB', label='True')

        # Predicted
        ax.plot(tgt_time, pred[:, f], '--', lw=2, color='#F18F01', label='Rollout')
        std = np.sqrt(var[:, f])
        ax.fill_between(tgt_time, pred[:, f] - 2*std, pred[:, f] + 2*std,
                        alpha=0.15, color='#F18F01')

        # Divider
        ax.axvline(CONTEXT_STEPS * 2, color='red', ls=':', lw=1, alpha=0.5)

        ax.set_title(FEATURE_NAMES[f], fontsize=11, fontweight='bold')
        ax.set_xlabel('Hours')
        ax.grid(True, alpha=0.3)
        if f == 0: ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, 'rollout_example.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================
# Print Summary
# ============================================================

def print_summary(metrics, history):
    """Print comprehensive text summary."""
    mdf = metrics
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Training: {len(history['train'])} epochs")
    print(f"  Final Train Loss: {history['train'][-1]:.6f}")
    print(f"  Final Test Loss:  {history['test'][-1]:.6f}")
    print(f"  Test Windows: {len(mdf)}")

    print(f"\n  Overall Metrics (masked, observed only):")
    print(f"    MSE:     {mdf['mse'].mean():.4f} (median: {mdf['mse'].median():.4f})")
    print(f"    MAE:     {mdf['mae'].mean():.4f} (median: {mdf['mae'].median():.4f})")
    print(f"    RMSE:    {mdf['rmse'].mean():.4f} (median: {mdf['rmse'].median():.4f})")
    print(f"    R²:      {mdf['r2'].mean():.4f} (median: {mdf['r2'].median():.4f})")
    print(f"    Pearson: {mdf['pearson'].mean():.4f} (median: {mdf['pearson'].median():.4f})")

    print(f"\n  Per-Feature (mean over test windows):")
    print(f"   {'Feature':<14} {'MSE':>8} {'MAE':>8} {'RMSE':>8} {'R²':>10} {'Type'}")
    print(f"   {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*6}")
    for f in range(INPUT_DIM):
        m = mdf[f'mse_f{f}'].mean()
        a = mdf[f'mae_f{f}'].mean()
        r = mdf[f'r2_f{f}'].mean()
        ft = 'Dense' if f in DENSE_INDICES else 'Sparse'
        print(f"   {FEATURE_NAMES[f]:<14} {m:8.4f} {a:8.4f} {np.sqrt(abs(m)):8.4f} {r:10.4f} [{ft}]")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize Digital Twin v5 Results')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing train.py outputs')
    args = parser.parse_args()

    history, metrics, predictions, ground_truth, target_masks, variances, \
        norm_params, rollout, config = load_results(args.results_dir)

    print(f"\n📊 Generating visualizations...")
    plot_overview(history, metrics, predictions, ground_truth, target_masks, config, args.results_dir)
    plot_per_feature(metrics, args.results_dir)
    plot_best_patient(metrics, predictions, ground_truth, target_masks,
                      variances, norm_params, args.results_dir)
    plot_loss_curves(history, args.results_dir)
    plot_rollout(rollout, norm_params, args.results_dir)

    print_summary(metrics, history)

    print(f"\n📄 Generated files in {args.results_dir}/:")
    print(f"   - overview.png")
    print(f"   - per_feature_metrics.png")
    print(f"   - best_patient_all_features.png")
    print(f"   - loss_curves.png")
    print(f"   - rollout_example.png")


if __name__ == "__main__":
    main()
