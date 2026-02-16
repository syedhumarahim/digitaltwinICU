"""
Digital Twin v5 — Training Script (Portable for DCC/GPU Cluster)
================================================================
Usage:
    python train.py --data_dir ./data --epochs 50 --batch_size 64

This script:
  1. Loads MIMIC-IV parquet data from data_dir
  2. Trains DigitalTwinSSM (Mamba + CDSP + Hawkes)
  3. Saves model, training history, and all test predictions
  4. Run visualize_results.py after training to generate plots
"""

import os
import glob
import gc
import random
import argparse
import json
import time as time_module
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from mamba_ssm import Mamba
from scipy import stats
from sklearn.metrics import r2_score

# ============================================================
# 1. Causal Modules
# ============================================================

class OrthogonalCDSP(nn.Module):
    """Numerically stable CDSP: || normalize(H)^T normalize(A) ||_F^2"""
    def __init__(self):
        super().__init__()

    def forward(self, h, a):
        if h.size(0) < 2:
            return torch.tensor(0.0, device=h.device)
        h_c = h - h.mean(dim=0, keepdim=True)
        a_c = a - a.mean(dim=0, keepdim=True)
        h_norm = h_c / (h_c.norm(dim=0, keepdim=True) + 1e-6)
        a_norm = a_c / (a_c.norm(dim=0, keepdim=True) + 1e-6)
        corr = torch.mm(h_norm.t(), a_norm)
        return torch.sum(corr ** 2) / (h.size(1) * a.size(1))


class DiscreteHawkesHead(nn.Module):
    """Latent intensity: λ_t = softplus(base + W·h_t)"""
    def __init__(self, hidden_dim, num_events):
        super().__init__()
        self.base_intensity = nn.Parameter(torch.zeros(num_events))
        self.history_proj = nn.Linear(hidden_dim, num_events)
        self.softplus = nn.Softplus()

    def forward(self, h_seq):
        return self.softplus(self.base_intensity + self.history_proj(h_seq))


# ============================================================
# 2. Digital Twin Model
# ============================================================

class DigitalTwinSSM(nn.Module):
    """
    Full-sequence Mamba SSM Digital Twin.
    Input: cat(x, mask, delta_t, a) → [B, T, input_dim*3 + treat_dim]
    ONE Mamba call on full sequence. h[t] predicts x[t+1].
    """
    def __init__(self, input_dim, treat_dim, hidden_dim, outcome_dim):
        super().__init__()
        self.input_dim = input_dim
        self.treat_dim = treat_dim
        self.hidden_dim = hidden_dim
        self.outcome_dim = outcome_dim

        total_input = input_dim * 3 + treat_dim
        self.input_proj = nn.Linear(total_input, hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim, d_state=16, d_conv=4)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, outcome_dim)
        self.logvar_head = nn.Linear(hidden_dim, outcome_dim)
        self.hawkes_head = DiscreteHawkesHead(hidden_dim, treat_dim)
        self.cdsp = OrthogonalCDSP()

    def forward(self, x_seq, mask_seq, dt_seq, a_seq):
        inp = torch.cat([x_seq, mask_seq, dt_seq, a_seq], dim=-1)
        proj = self.input_proj(inp)
        h_seq = self.mamba(proj)
        h_seq = self.norm(h_seq)

        if not hasattr(self, '_shape_printed'):
            print(f"  ✓ Mamba input shape:  {proj.shape}  (full sequence)")
            print(f"  ✓ Mamba output shape: {h_seq.shape}")
            self._shape_printed = True

        y_pred = self.mean_head(h_seq)
        y_var = F.softplus(self.logvar_head(h_seq)) + 1e-4
        hawkes = self.hawkes_head(h_seq)

        cdsp_loss = torch.tensor(0.0, device=x_seq.device)
        if self.training:
            h_sub = h_seq[:, ::4, :]
            a_sub = a_seq[:, ::4, :]
            cdsp_loss = self.cdsp(h_sub.reshape(-1, self.hidden_dim),
                                  a_sub.reshape(-1, self.treat_dim))

        return y_pred, y_var, hawkes, cdsp_loss, h_seq

    def rollout(self, x_ctx, mask_ctx, dt_ctx, a_ctx, a_future, horizon):
        """Autoregressive rollout using FULL context warmup."""
        B, ctx_len, D = x_ctx.shape
        device = x_ctx.device

        inp_ctx = torch.cat([x_ctx, mask_ctx, dt_ctx, a_ctx], dim=-1)
        proj_full = self.input_proj(inp_ctx)

        preds, variances = [], []
        for h in range(horizon):
            h_seq = self.norm(self.mamba(proj_full))
            pred = self.mean_head(h_seq[:, -1:])
            var = F.softplus(self.logvar_head(h_seq[:, -1:])) + 1e-4
            preds.append(pred)
            variances.append(var)

            next_x = pred.detach()
            next_mask = torch.ones(B, 1, D, device=device)
            next_dt = torch.zeros(B, 1, D, device=device)
            next_a = a_future[:, h:h+1]
            next_inp = torch.cat([next_x, next_mask, next_dt, next_a], dim=-1)
            proj_full = torch.cat([proj_full, self.input_proj(next_inp)], dim=1)

        return torch.cat(preds, dim=1), torch.cat(variances, dim=1)


# ============================================================
# 3. Feature Configuration
# ============================================================

VITAL_FEATURES = [
    'LAB//220045', 'LAB//220277', 'LAB//220181', 'LAB//220210',
    'LAB//223762', 'LAB//50912', 'LAB//50813', 'LAB//220052',
    'LAB//50971', 'LAB//50983', 'LAB//50902', 'LAB//51006',
    'LAB//50931', 'LAB//51265', 'LAB//51222',
]
DENSE_INDICES = [0, 1, 2, 3]
SPARSE_INDICES = [4,5,6,7,8,9,10,11,12,13,14]

TREATMENT_FEATURES = [
    'MEDICATION//START//0', 'MEDICATION//STOP//0',
    'LAB//225158', 'LAB//224082', 'LAB//224084',
]

FEATURE_NAMES = [
    'Heart Rate', 'O2 Sat', 'NBP Mean', 'Resp Rate', 'Temp',
    'Creatinine', 'Lactate', 'MAP', 'Potassium', 'Sodium',
    'Chloride', 'BUN', 'Glucose', 'Platelets', 'Hemoglobin',
]
TREATMENT_NAMES = ['Med Start', 'Med Stop', 'IV NaCl', 'Turn', 'Intervention']

INPUT_DIM = 15
TREAT_DIM = 5
RESAMPLE_FREQ = '2h'
CONTEXT_HOURS = 72
HORIZON_HOURS = 24
CONTEXT_STEPS = 36
HORIZON_STEPS = 12
TOTAL_STEPS = 48
STRIDE_STEPS = 1


# ============================================================
# 4. Dataset and Collation
# ============================================================

class ICUWindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        return self.windows[idx]

def collate_fn(batch):
    keys = ['x_full', 'mask_full', 'dt_full', 'a_full', 'target_mask']
    padded = {}
    for k in keys:
        padded[k] = pad_sequence([item[k] for item in batch], batch_first=True, padding_value=0.0)
    padded['lengths'] = torch.tensor([len(item['x_full']) for item in batch])
    return padded


# ============================================================
# 5. Data Loading
# ============================================================

def load_data(data_dir, target_subjects=None):
    """
    Load MIMIC-IV parquet data.

    Args:
        data_dir: directory containing parquet files and icustays2.csv
        target_subjects: max subjects to load (None = ALL)
    """
    parquet_dir = '/hpc/group/kamaleswaranlab/capstone_icu_digital_twins/meds/MIMIC-IV_Example/data/MEDS_COHORT/data/train'
    icustays_path = '/hpc/group/kamaleswaranlab/state-space-Digital-Twin/icustays2.csv'

    files = sorted(glob.glob(os.path.join(parquet_dir, '*.parquet')))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

    label = f"{target_subjects}" if target_subjects else "ALL"
    print(f"{'='*70}")
    print(f"DIGITAL TWIN v5 — DATA LOADING ({label} subjects)")
    print(f"{'='*70}")
    print(f"  Data dir: {data_dir}")
    print(f"  Parquet files: {len(files)}")
    print(f"  Resampling: {RESAMPLE_FREQ} bins using .last()")
    print(f"  Window: {CONTEXT_STEPS} ctx + {HORIZON_STEPS} tgt = {TOTAL_STEPS} steps")
    print(f"  Input dims: {INPUT_DIM}*3 + {TREAT_DIM} = {INPUT_DIM*3+TREAT_DIM}")

    icu_stays = pd.read_csv(icustays_path)
    icu_stays['intime'] = pd.to_datetime(icu_stays['intime'])
    icu_stays['outtime'] = pd.to_datetime(icu_stays['outtime'])
    icu_subject_ids = set(icu_stays['subject_id'].unique())
    print(f"  ICU stays: {len(icu_stays)} from {len(icu_subject_ids)} patients")

    # Load events
    print(f"\n1⃣  Loading subjects...")
    collected_dfs = []
    unique_subjects = set()

    for f in files:
        if target_subjects and len(unique_subjects) >= target_subjects:
            break
        df = pd.read_parquet(f)
        df = df[df['code'].isin(VITAL_FEATURES + TREATMENT_FEATURES)]
        df = df[df['subject_id'].isin(icu_subject_ids)]
        collected_dfs.append(df)
        unique_subjects.update(df['subject_id'].unique())
        if len(collected_dfs) % 10 == 0:
            print(f"   ... {len(collected_dfs)} files, {len(unique_subjects)} subjects")

    df_all = pd.concat(collected_dfs, ignore_index=True)
    df_all['time'] = pd.to_datetime(df_all['time'])

    if target_subjects:
        final_subjects = list(unique_subjects)[:target_subjects]
    else:
        final_subjects = list(unique_subjects)
    df_all = df_all[df_all['subject_id'].isin(final_subjects)]
    print(f"   ✓ {len(final_subjects)} subjects, {len(df_all):,} events")

    # Pivot
    df_numeric = df_all[df_all['code'].isin(VITAL_FEATURES)]
    df_events = df_all[df_all['code'].isin(TREATMENT_FEATURES)]

    df_x = df_numeric.pivot_table(index=['subject_id', 'time'],
                                   columns='code', values='numeric_value')
    df_events_copy = df_events.copy()
    df_events_copy['event'] = 1
    df_a = df_events_copy.pivot_table(index=['subject_id', 'time'],
                                       columns='code', values='event', aggfunc='max')

    del df_all, df_numeric, df_events, df_events_copy, collected_dfs
    gc.collect()

    # Process patients → windows
    print(f"\n2⃣  Processing patients → windows...")
    all_windows = []
    patients_processed = 0
    patients_skipped = 0
    MAX_STEPS = 100

    for sub in final_subjects:
        if sub not in df_x.index:
            patients_skipped += 1
            continue

        sub_stays = icu_stays[icu_stays['subject_id'] == sub]
        if sub_stays.empty:
            patients_skipped += 1
            continue

        longest = sub_stays.loc[sub_stays['los'].idxmax()]
        icu_in, icu_out = longest['intime'], longest['outtime']

        sub_data = df_x.loc[sub]
        sub_data = sub_data[(sub_data.index >= icu_in) & (sub_data.index <= icu_out)]
        if sub_data.empty:
            patients_skipped += 1
            continue

        x_raw = sub_data.resample(RESAMPLE_FREQ).last()
        x_raw = x_raw.reindex(columns=VITAL_FEATURES, fill_value=np.nan)

        if len(x_raw) > MAX_STEPS:
            x_raw = x_raw.iloc[:MAX_STEPS]
        if len(x_raw) < TOTAL_STEPS:
            patients_skipped += 1
            continue

        obs_mask = (~x_raw.isna()).astype(float)

        # Delta_t
        delta_t = pd.DataFrame(0.0, index=x_raw.index, columns=VITAL_FEATURES)
        for col in VITAL_FEATURES:
            last_obs_idx = None
            dt_col = []
            for idx, val in enumerate(x_raw[col].values):
                if pd.notna(val):
                    dt_col.append(0.0)
                    last_obs_idx = idx
                else:
                    dt_col.append(float(idx - last_obs_idx) if last_obs_idx is not None else 999.0)
            delta_t[col] = dt_col

        # Per-patient normalization
        x_mean = x_raw.mean(skipna=True)
        x_std = x_raw.std(skipna=True) + 1e-6
        x_norm = (x_raw - x_mean) / x_std
        x_norm_filled = x_norm.fillna(0.0)

        # Treatments
        if sub in df_a.index:
            a = df_a.loc[sub].resample(RESAMPLE_FREQ).max().fillna(0)
            a = a.reindex(x_raw.index, fill_value=0)
        else:
            a = pd.DataFrame(0, index=x_raw.index, columns=TREATMENT_FEATURES)
        a = a.reindex(columns=TREATMENT_FEATURES, fill_value=0)

        # Tensors
        x_raw_filled = x_raw.fillna(0.0)
        x_raw_tensor = torch.tensor(x_raw_filled.values, dtype=torch.float32)
        x_tensor = torch.tensor(x_norm_filled.values, dtype=torch.float32)
        mask_tensor = torch.tensor(obs_mask.values, dtype=torch.float32)
        dt_tensor = torch.tensor(delta_t.values, dtype=torch.float32).clamp(0, 999)
        a_tensor = torch.tensor(a.values, dtype=torch.float32)

        # Windows
        T = len(x_tensor)
        for start in range(0, T - TOTAL_STEPS + 1, STRIDE_STEPS):
            end = start + TOTAL_STEPS
            ctx_end = start + CONTEXT_STEPS
            tgt_mask = mask_tensor[ctx_end:end]
            if tgt_mask.sum() < 3:
                continue

            window = {
                'x_full': x_tensor[start:end],
                'mask_full': mask_tensor[start:end],
                'dt_full': dt_tensor[start:end],
                'a_full': a_tensor[start:end],
                'target_mask': tgt_mask,
                'x_raw_full': x_raw_tensor[start:end],
                'subject_id': sub,
                'norm_params': {'mean': x_mean.values, 'std': x_std.values},
            }
            if not torch.isnan(window['x_full']).any():
                all_windows.append(window)

        patients_processed += 1
        if patients_processed % 100 == 0:
            print(f"   ... {patients_processed} patients → {len(all_windows)} windows")

    # Stats
    avg_obs = 0.0
    if all_windows:
        avg_obs = np.mean([w['target_mask'].sum().item() / (HORIZON_STEPS * INPUT_DIM) * 100
                           for w in all_windows])

    print(f"\n   ✓ Processed: {patients_processed} patients")
    print(f"   ✓ Skipped: {patients_skipped}")
    print(f"   ✓ Windows: {len(all_windows):,}")
    print(f"   ✓ Avg target obs: {avg_obs:.1f}%")
    return all_windows


# ============================================================
# 6. Loss Functions
# ============================================================

def masked_loss_dense_sparse(pred, target, target_mask, alpha=1.0):
    sq_err = (pred - target) ** 2
    d_mask = target_mask[:, :, DENSE_INDICES]
    L_v = (sq_err[:, :, DENSE_INDICES] * d_mask).sum() / (d_mask.sum() + 1e-8)
    s_mask = target_mask[:, :, SPARSE_INDICES]
    L_l = (sq_err[:, :, SPARSE_INDICES] * s_mask).sum() / (s_mask.sum() + 1e-8)
    return L_v + alpha * L_l, L_v.item(), L_l.item()

def hawkes_nll(intensities, events):
    return (intensities - events * torch.log(intensities + 1e-6)).mean()


# ============================================================
# 7. Training
# ============================================================

def train(train_windows, test_windows, device, args):
    """Main training loop."""
    model = DigitalTwinSSM(INPUT_DIM, TREAT_DIM, args.hidden_dim, INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_loader = DataLoader(ICUWindowDataset(train_windows), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(ICUWindowDataset(test_windows), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"\n{'='*70}")
    print(f"TRAINING — Digital Twin v5")
    print(f"{'='*70}")
    print(f"  Model: Mamba (d={args.hidden_dim}, state=16, conv=4)")
    print(f"  Input: {INPUT_DIM*3+TREAT_DIM}d, Sequence: {TOTAL_STEPS} steps")
    print(f"  Loss: L_recon + {args.lambda_cdsp}·CDSP + {args.lambda_hawkes}·Hawkes")
    print(f"  CDSP warmup: {args.cdsp_warmup} epochs")
    print(f"  Batch: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")
    print(f"  Train: {len(train_windows):,}, Test: {len(test_windows):,}")
    print(f"  Device: {device}")

    history = {'train': [], 'test': [], 'vital': [], 'lab': [], 'cdsp': [], 'hawkes': []}
    best_test = float('inf')

    for epoch in range(args.epochs):
        t0 = time_module.time()

        # --- TRAIN ---
        model.train()
        e_loss = e_vit = e_lab = e_cdsp = e_hawk = 0.0
        n_batch = 0
        cdsp_w = 0.0 if epoch < args.cdsp_warmup else args.lambda_cdsp

        for batch in train_loader:
            x = batch['x_full'].to(device)
            mask = batch['mask_full'].to(device)
            dt = batch['dt_full'].to(device)
            a = batch['a_full'].to(device)
            tm = batch['target_mask'].to(device)

            y_pred, y_var, hawkes_int, cdsp_loss, h_seq = model(x, mask, dt, a)

            pred_tgt = y_pred[:, CONTEXT_STEPS-1:CONTEXT_STEPS+HORIZON_STEPS-1]
            true_tgt = x[:, CONTEXT_STEPS:CONTEXT_STEPS+HORIZON_STEPS]

            loss_r, l_v, l_l = masked_loss_dense_sparse(pred_tgt, true_tgt, tm, args.alpha)
            loss_c = cdsp_w * cdsp_loss
            loss_h = args.lambda_hawkes * hawkes_nll(hawkes_int, a)
            total = loss_r + loss_c + loss_h

            if torch.isnan(total):
                print(f"  ⚠ NaN at epoch {epoch+1}, skipping")
                continue

            optimizer.zero_grad()
            total.backward()

            if epoch == 0 and n_batch == 0:
                has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                             for p in model.parameters())
                print(f"  ✓ Gradient check: {'⚠ NaN!' if has_nan else 'PASS'}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            e_loss += loss_r.item()
            e_vit += l_v; e_lab += l_l
            e_cdsp += cdsp_loss.item() if torch.is_tensor(cdsp_loss) else cdsp_loss
            e_hawk += loss_h.item()
            n_batch += 1

        scheduler.step()
        nb = max(n_batch, 1)
        history['train'].append(e_loss/nb)
        history['vital'].append(e_vit/nb)
        history['lab'].append(e_lab/nb)
        history['cdsp'].append(e_cdsp/nb)
        history['hawkes'].append(e_hawk/nb)

        # --- TEST ---
        model.eval()
        t_loss, t_batch = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x_full'].to(device)
                mask = batch['mask_full'].to(device)
                dt = batch['dt_full'].to(device)
                a = batch['a_full'].to(device)
                tm = batch['target_mask'].to(device)

                y_pred, _, _, _, _ = model(x, mask, dt, a)
                pred_tgt = y_pred[:, CONTEXT_STEPS-1:CONTEXT_STEPS+HORIZON_STEPS-1]
                true_tgt = x[:, CONTEXT_STEPS:CONTEXT_STEPS+HORIZON_STEPS]
                loss, _, _ = masked_loss_dense_sparse(pred_tgt, true_tgt, tm, args.alpha)
                t_loss += loss.item()
                t_batch += 1

        test_avg = t_loss / max(t_batch, 1)
        history['test'].append(test_avg)

        # Save best model
        if test_avg < best_test:
            best_test = test_avg
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))

        elapsed = time_module.time() - t0
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  E{epoch+1:3d}/{args.epochs} | Recon: {e_loss/nb:.4f} "
                  f"(V:{e_vit/nb:.4f} L:{e_lab/nb:.4f}) "
                  f"CDSP:{e_cdsp/nb:.5f} Hawk:{e_hawk/nb:.4f} "
                  f"| Test: {test_avg:.4f} | {elapsed:.1f}s")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))

    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    print(f"\n  ✓ Best test loss: {best_test:.6f}")
    print(f"  ✓ Models saved to {args.output_dir}/")

    return model, history, test_loader, test_windows


# ============================================================
# 8. Post-Training: Compute + Save All Predictions
# ============================================================

def save_predictions(model, test_loader, test_windows, device, output_dir):
    """Compute all test predictions and save for visualization."""
    print("\n📊 Computing test predictions...")
    model.eval()
    model.to(device)

    all_preds = []
    all_true = []
    all_masks = []
    all_vars = []
    all_norm_params = []
    all_raw = []
    win_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x_full'].to(device)
            mask = batch['mask_full'].to(device)
            dt = batch['dt_full'].to(device)
            a = batch['a_full'].to(device)
            tm = batch['target_mask'].to(device)

            y_pred, y_var, _, _, _ = model(x, mask, dt, a)
            pred_tgt = y_pred[:, CONTEXT_STEPS-1:CONTEXT_STEPS+HORIZON_STEPS-1]
            true_tgt = x[:, CONTEXT_STEPS:CONTEXT_STEPS+HORIZON_STEPS]

            B = x.shape[0]
            for i in range(B):
                all_preds.append(pred_tgt[i].cpu().numpy())
                all_true.append(true_tgt[i].cpu().numpy())
                all_masks.append(tm[i].cpu().numpy())
                all_vars.append(y_var[i, CONTEXT_STEPS-1:CONTEXT_STEPS+HORIZON_STEPS-1].cpu().numpy())

                if win_idx < len(test_windows):
                    w = test_windows[win_idx]
                    all_norm_params.append({
                        'mean': w['norm_params']['mean'].tolist(),
                        'std': w['norm_params']['std'].tolist()
                    })
                    all_raw.append(w['x_raw_full'].numpy().tolist())
                else:
                    all_norm_params.append(None)
                    all_raw.append(None)
                win_idx += 1

    # Save as numpy arrays
    np.savez_compressed(
        os.path.join(output_dir, 'test_predictions.npz'),
        predictions=np.array(all_preds),
        ground_truth=np.array(all_true),
        target_masks=np.array(all_masks),
        variances=np.array(all_vars),
    )

    # Save norm params
    with open(os.path.join(output_dir, 'norm_params.json'), 'w') as f:
        json.dump(all_norm_params, f)

    # Rollout example
    if test_windows:
        w0 = test_windows[0]
        x_ctx = w0['x_full'][:CONTEXT_STEPS].unsqueeze(0).to(device)
        m_ctx = w0['mask_full'][:CONTEXT_STEPS].unsqueeze(0).to(device)
        d_ctx = w0['dt_full'][:CONTEXT_STEPS].unsqueeze(0).to(device)
        a_ctx = w0['a_full'][:CONTEXT_STEPS].unsqueeze(0).to(device)
        a_fut = w0['a_full'][CONTEXT_STEPS:].unsqueeze(0).to(device)

        rollout_pred, rollout_var = model.rollout(x_ctx, m_ctx, d_ctx, a_ctx, a_fut, HORIZON_STEPS)
        np.savez(os.path.join(output_dir, 'rollout_example.npz'),
                 prediction=rollout_pred.cpu().detach().numpy(),
                 variance=rollout_var.cpu().detach().numpy(),
                 context=w0['x_full'][:CONTEXT_STEPS].numpy(),
                 target=w0['x_full'][CONTEXT_STEPS:].numpy(),
                 target_mask=w0['target_mask'].numpy())
        print(f"  ✓ Rollout: {x_ctx.shape} → {rollout_pred.shape}")

    # Compute per-window metrics
    metrics_rows = []
    for i in range(len(all_preds)):
        pi, ti, mi = all_preds[i], all_true[i], all_masks[i]
        obs = mi > 0.5
        if obs.sum() < 3:
            continue
        po, to_ = pi[obs], ti[obs]
        mse = np.mean((po - to_)**2)
        mae = np.mean(np.abs(po - to_))
        try: r2 = r2_score(to_, po)
        except: r2 = 0.0
        try:
            pear = stats.pearsonr(to_, po)[0]
            if np.isnan(pear): pear = 0.0
        except: pear = 0.0

        row = {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse), 'r2': r2, 'pearson': pear}
        for f in range(INPUT_DIM):
            fm = mi[:, f] > 0.5
            if fm.sum() < 2:
                row[f'mse_f{f}'] = row[f'mae_f{f}'] = row[f'r2_f{f}'] = np.nan
            else:
                row[f'mse_f{f}'] = np.mean((pi[fm,f]-ti[fm,f])**2)
                row[f'mae_f{f}'] = np.mean(np.abs(pi[fm,f]-ti[fm,f]))
                try: row[f'r2_f{f}'] = r2_score(ti[fm,f], pi[fm,f])
                except: row[f'r2_f{f}'] = 0.0
        metrics_rows.append(row)

    mdf = pd.DataFrame(metrics_rows)
    mdf.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    print(f"  ✓ Saved {len(all_preds)} predictions to {output_dir}/")
    print(f"  ✓ Metrics: MSE={mdf['mse'].mean():.4f}, R²={mdf['r2'].mean():.4f}")

    # Print per-feature summary
    print(f"\n  Per-Feature:")
    print(f"   {'Feature':<14} {'MSE':>8} {'MAE':>8} {'RMSE':>8} {'R²':>10} {'Type'}")
    for f in range(INPUT_DIM):
        m = mdf[f'mse_f{f}'].mean()
        a = mdf[f'mae_f{f}'].mean()
        r = mdf[f'r2_f{f}'].mean()
        ft = 'Dense' if f in DENSE_INDICES else 'Sparse'
        print(f"   {FEATURE_NAMES[f]:<14} {m:8.4f} {a:8.4f} {np.sqrt(m):8.4f} {r:10.4f} [{ft}]")


# ============================================================
# 9. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Digital Twin v5 Training')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory with mimic_iv_parquet_files/ and icustays2.csv')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--subjects', type=int, default=None,
                        help='Max subjects (None=all, 1307=2%%)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Lab loss weight')
    parser.add_argument('--lambda_cdsp', type=float, default=0.1)
    parser.add_argument('--lambda_hawkes', type=float, default=0.01)
    parser.add_argument('--cdsp_warmup', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🚀 Device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"⚠️  Device: CPU")

    # Load
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_windows = load_data(args.data_dir, args.subjects)
    if not all_windows:
        print("ERROR: No windows extracted.")
        return

    random.shuffle(all_windows)
    split = int(len(all_windows) * 0.8)
    train_w, test_w = all_windows[:split], all_windows[split:]
    print(f"\n📊 Split: {len(train_w):,} train, {len(test_w):,} test")

    # Train
    model, history, test_loader, test_windows = train(train_w, test_w, device, args)

    # Save predictions
    save_predictions(model, test_loader, test_w, device, args.output_dir)

    print(f"\n{'='*70}")
    print(f"✅ DONE — Results in {args.output_dir}/")
    print(f"   Run: python visualize_results.py --results_dir {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
