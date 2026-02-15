"""
Digital Twin v5 — Final Consolidated Implementation
=====================================================
Architecture:
  - Full-sequence Mamba SSM (ONE call, no timestep loop)
  - OrthogonalCDSP decorrelation penalty
  - DiscreteHawkesHead intensity regularization
  - Masked loss (dense vitals / sparse labs)
  - Teacher-forced training, autoregressive rollout
  - 2h aggregation with .last(), no LOV, no imputation
  - Input: x + mask + delta_t + a = 15*3 + 5 = 50 dims
"""

import os
import glob
import gc
import random
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100

# ============================================================
# 1. Causal Modules
# ============================================================

class OrthogonalCDSP(nn.Module):
    """
    Numerically stable CDSP: penalizes cosine similarity between
    hidden states and treatment indicators.
    L_cdsp = || normalize(H)^T normalize(A) ||_F^2
    """
    def __init__(self):
        super().__init__()

    def forward(self, h, a):
        """h: [N, hidden_dim], a: [N, treat_dim]"""
        if h.size(0) < 2:
            return torch.tensor(0.0, device=h.device)

        # Center
        h_c = h - h.mean(dim=0, keepdim=True)
        a_c = a - a.mean(dim=0, keepdim=True)

        # Normalize columns (epsilon for stability)
        h_norm = h_c / (h_c.norm(dim=0, keepdim=True) + 1e-6)
        a_norm = a_c / (a_c.norm(dim=0, keepdim=True) + 1e-6)

        # Correlation matrix: (hidden, N) @ (N, treat) → (hidden, treat)
        corr = torch.mm(h_norm.t(), a_norm)

        # Frobenius norm squared, scaled
        return torch.sum(corr ** 2) / (h.size(1) * a.size(1))


class DiscreteHawkesHead(nn.Module):
    """
    Latent conditional intensity regularization.
    λ_t = softplus(base + W·h_t)
    Loss: λ_t - k_t · log(λ_t + ε)  (Poisson NLL)
    """
    def __init__(self, hidden_dim, num_events):
        super().__init__()
        self.base_intensity = nn.Parameter(torch.zeros(num_events))
        self.history_proj = nn.Linear(hidden_dim, num_events)
        self.softplus = nn.Softplus()

    def forward(self, h_seq):
        """h_seq: [B, T, hidden_dim] → λ: [B, T, num_events]"""
        return self.softplus(self.base_intensity + self.history_proj(h_seq))


# ============================================================
# 2. Digital Twin Model — Full-Sequence SSM
# ============================================================

class DigitalTwinSSM(nn.Module):
    """
    Selective State-Space Digital Twin.

    Input per timestep: cat(x_t, mask_t, delta_t_t, a_t)
      = input_dim*3 + treat_dim = 50

    Single Mamba call on full sequence → decode predictions.
    h[t] predicts x[t+1].
    """
    def __init__(self, input_dim, treat_dim, hidden_dim, outcome_dim):
        super().__init__()
        self.input_dim = input_dim
        self.treat_dim = treat_dim
        self.hidden_dim = hidden_dim
        self.outcome_dim = outcome_dim

        total_input = input_dim * 3 + treat_dim  # x + mask + dt + a
        self.input_proj = nn.Linear(total_input, hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim, d_state=16, d_conv=4)
        self.norm = nn.LayerNorm(hidden_dim)

        # Prediction heads
        self.mean_head = nn.Linear(hidden_dim, outcome_dim)
        self.logvar_head = nn.Linear(hidden_dim, outcome_dim)

        # Causal modules
        self.hawkes_head = DiscreteHawkesHead(hidden_dim, treat_dim)
        self.cdsp = OrthogonalCDSP()

    def forward(self, x_seq, mask_seq, dt_seq, a_seq):
        """
        Full-sequence forward pass (teacher-forced training).

        Args: all [B, T, D] or [B, T, treat_dim]
        Returns:
            y_pred:    [B, T, outcome_dim]
            y_var:     [B, T, outcome_dim]  (softplus, not exp)
            hawkes:    [B, T, treat_dim]
            cdsp_loss: scalar
            h_seq:     [B, T, hidden_dim]
        """
        # Concatenate input channels
        inp = torch.cat([x_seq, mask_seq, dt_seq, a_seq], dim=-1)  # [B, T, 50]

        # Project and process through Mamba — ONE CALL
        proj = self.input_proj(inp)          # [B, T, hidden]
        h_seq = self.mamba(proj)             # [B, T, hidden] — true selective scan
        h_seq = self.norm(h_seq)

        # Verify shape (printed once)
        if not hasattr(self, '_shape_printed'):
            print(f"  ✓ Mamba input shape:  {proj.shape}  (full sequence)")
            print(f"  ✓ Mamba output shape: {h_seq.shape}")
            self._shape_printed = True

        # Decode
        y_pred = self.mean_head(h_seq)                        # [B, T, D]
        y_var = F.softplus(self.logvar_head(h_seq)) + 1e-4    # bounded variance

        # Hawkes intensity
        hawkes = self.hawkes_head(h_seq)                      # [B, T, treat_dim]

        # CDSP penalty (training only, on subsampled steps)
        cdsp_loss = torch.tensor(0.0, device=x_seq.device)
        if self.training:
            # Subsample every 4 steps, flatten to [B*T_sub, D]
            h_sub = h_seq[:, ::4, :]
            a_sub = a_seq[:, ::4, :]
            h_flat = h_sub.reshape(-1, self.hidden_dim)
            a_flat = a_sub.reshape(-1, self.treat_dim)
            cdsp_loss = self.cdsp(h_flat, a_flat)

        return y_pred, y_var, hawkes, cdsp_loss, h_seq

    def rollout(self, x_ctx, mask_ctx, dt_ctx, a_ctx, a_future, horizon):
        """
        Autoregressive rollout for simulation.
        Uses FULL context for warmup (no truncation).

        Args:
            x_ctx:     [B, ctx_len, D]  — context observations
            mask_ctx:  [B, ctx_len, D]
            dt_ctx:    [B, ctx_len, D]
            a_ctx:     [B, ctx_len, treat_dim]
            a_future:  [B, horizon, treat_dim]  — future treatment schedule
            horizon:   int — number of steps to predict

        Returns:
            preds:     [B, horizon, D]
            variances: [B, horizon, D]
        """
        B, ctx_len, D = x_ctx.shape
        device = x_ctx.device

        # Build projected context — FULL warmup
        inp_ctx = torch.cat([x_ctx, mask_ctx, dt_ctx, a_ctx], dim=-1)
        proj_full = self.input_proj(inp_ctx)  # [B, ctx_len, hidden]

        preds = []
        variances = []

        for h in range(horizon):
            # Forward full projected sequence through Mamba (causal)
            h_seq = self.norm(self.mamba(proj_full))

            # Predict from last position
            pred = self.mean_head(h_seq[:, -1:])              # [B, 1, D]
            var = F.softplus(self.logvar_head(h_seq[:, -1:])) + 1e-4
            preds.append(pred)
            variances.append(var)

            # Build next input from prediction (no gradient for stability)
            next_x = pred.detach()
            next_mask = torch.ones(B, 1, D, device=device)
            next_dt = torch.zeros(B, 1, D, device=device)
            next_a = a_future[:, h:h+1]

            next_inp = torch.cat([next_x, next_mask, next_dt, next_a], dim=-1)
            next_proj = self.input_proj(next_inp)

            # Extend the projected sequence
            proj_full = torch.cat([proj_full, next_proj], dim=1)

        return torch.cat(preds, dim=1), torch.cat(variances, dim=1)


# ============================================================
# 3. Feature Configuration
# ============================================================

VITAL_FEATURES = [
    'LAB//220045',   # Heart Rate
    'LAB//220277',   # O2 Saturation
    'LAB//220181',   # Non-Invasive BP Mean
    'LAB//220210',   # Respiratory Rate
    'LAB//223762',   # Temperature (Celsius)
    'LAB//50912',    # Creatinine
    'LAB//50813',    # Lactate
    'LAB//220052',   # Arterial BP Mean (MAP)
    'LAB//50971',    # Potassium
    'LAB//50983',    # Sodium
    'LAB//50902',    # Chloride
    'LAB//51006',    # BUN
    'LAB//50931',    # Glucose (Blood)
    'LAB//51265',    # Platelets
    'LAB//51222',    # Hemoglobin
]

DENSE_INDICES = [0, 1, 2, 3]           # HR, O2, NBP, RR (~80%+ at 2h)
SPARSE_INDICES = [4,5,6,7,8,9,10,11,12,13,14]  # Temp + labs

TREATMENT_FEATURES = [
    'MEDICATION//START//0',
    'MEDICATION//STOP//0',
    'LAB//225158',
    'LAB//224082',
    'LAB//224084',
]

FEATURE_NAMES = [
    'Heart Rate', 'O2 Sat', 'NBP Mean', 'Resp Rate', 'Temp',
    'Creatinine', 'Lactate', 'MAP', 'Potassium', 'Sodium',
    'Chloride', 'BUN', 'Glucose', 'Platelets', 'Hemoglobin',
]
TREATMENT_NAMES = ['Med Start', 'Med Stop', 'IV NaCl', 'Turn Patient', 'Intervention']

INPUT_DIM = 15
TREAT_DIM = 5

# Temporal parameters (2h bins)
RESAMPLE_FREQ = '2h'
CONTEXT_HOURS = 72
HORIZON_HOURS = 24
CONTEXT_STEPS = CONTEXT_HOURS // 2   # 36
HORIZON_STEPS = HORIZON_HOURS // 2   # 12
TOTAL_STEPS = CONTEXT_STEPS + HORIZON_STEPS  # 48
STRIDE_STEPS = 1  # = 2h stride


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
    """Collate full-sequence windows with target masks."""
    keys = ['x_full', 'mask_full', 'dt_full', 'a_full', 'target_mask']
    padded = {}
    for k in keys:
        seqs = [item[k] for item in batch]
        padded[k] = pad_sequence(seqs, batch_first=True, padding_value=0.0)

    lengths = torch.tensor([len(item['x_full']) for item in batch])
    return {**padded, 'lengths': lengths}


# ============================================================
# 5. Data Loading Pipeline
# ============================================================

def load_data_v5(target_subjects=200):
    """
    Final data pipeline:
      - 2h aggregation with .last()
      - No LOV, no imputation
      - Full-sequence windows (context + target)
      - Target mask for observed-only loss
    """
    files = sorted(glob.glob('/Users/syedhumashah/Downloads/mimic_iv_parquet_files/*.parquet'))
    if not files:
        raise FileNotFoundError("No .parquet files found.")

    print(f"{'='*70}")
    print("DIGITAL TWIN v5 — DATA LOADING")
    print(f"{'='*70}")
    print(f"  Features: {INPUT_DIM} vitals/labs + {TREAT_DIM} treatments")
    print(f"  Resampling: {RESAMPLE_FREQ} bins using .last()")
    print(f"  Window: {CONTEXT_STEPS} ctx + {HORIZON_STEPS} tgt = {TOTAL_STEPS} steps")
    print(f"  Input dims: {INPUT_DIM}*3 + {TREAT_DIM} = {INPUT_DIM*3+TREAT_DIM}")
    print(f"  No LOV, no imputation, no delta prediction")

    ICUSTAYS_PATH = '/Users/syedhumashah/Downloads/icustays2.csv'
    icu_stays = pd.read_csv(ICUSTAYS_PATH)
    icu_stays['intime'] = pd.to_datetime(icu_stays['intime'])
    icu_stays['outtime'] = pd.to_datetime(icu_stays['outtime'])
    icu_subject_ids = set(icu_stays['subject_id'].unique())
    print(f"  ICU stays: {len(icu_stays)} from {len(icu_subject_ids)} patients")

    print(f"\n1⃣  Loading up to {target_subjects} subjects...")
    collected_dfs = []
    unique_subjects = set()

    for f in files:
        if len(unique_subjects) >= target_subjects:
            break
        df_chunk = pd.read_parquet(f)
        mask = df_chunk['code'].isin(VITAL_FEATURES + TREATMENT_FEATURES)
        df_chunk = df_chunk[mask]
        df_chunk = df_chunk[df_chunk['subject_id'].isin(icu_subject_ids)]
        collected_dfs.append(df_chunk)
        unique_subjects.update(df_chunk['subject_id'].unique())

    df_all = pd.concat(collected_dfs, ignore_index=True)
    df_all['time'] = pd.to_datetime(df_all['time'])
    final_subjects = list(unique_subjects)[:target_subjects]
    df_all = df_all[df_all['subject_id'].isin(final_subjects)]

    print(f"   ✓ {len(final_subjects)} subjects, {len(df_all):,} events")

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

    print(f"\n2⃣  Processing patients → full-sequence windows...")
    all_windows = []
    patients_processed = 0
    patients_skipped = 0
    MAX_STEPS = 100  # Max 200h / 2h = 100 steps

    for sub in final_subjects:
        if sub not in df_x.index:
            patients_skipped += 1
            continue

        sub_stays = icu_stays[icu_stays['subject_id'] == sub]
        if sub_stays.empty:
            patients_skipped += 1
            continue

        longest_stay = sub_stays.loc[sub_stays['los'].idxmax()]
        icu_in, icu_out = longest_stay['intime'], longest_stay['outtime']

        sub_data = df_x.loc[sub]
        sub_data = sub_data[(sub_data.index >= icu_in) & (sub_data.index <= icu_out)]
        if sub_data.empty:
            patients_skipped += 1
            continue

        # --- 2h aggregation with .last() ---
        x_raw = sub_data.resample(RESAMPLE_FREQ).last()
        x_raw = x_raw.reindex(columns=VITAL_FEATURES, fill_value=np.nan)

        if len(x_raw) > MAX_STEPS:
            x_raw = x_raw.iloc[:MAX_STEPS]

        if len(x_raw) < TOTAL_STEPS:
            patients_skipped += 1
            continue

        # --- Observation mask ---
        obs_mask = (~x_raw.isna()).astype(float)

        # --- Delta_t: steps since last observation per feature ---
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

        # --- Per-patient normalization (observed only) ---
        x_mean = x_raw.mean(skipna=True)
        x_std = x_raw.std(skipna=True) + 1e-6

        x_norm = (x_raw - x_mean) / x_std
        x_norm_filled = x_norm.fillna(0.0)  # NaN → 0 (patient's mean)

        # --- Treatments ---
        if sub in df_a.index:
            a = df_a.loc[sub].resample(RESAMPLE_FREQ).max().fillna(0)
            a = a.reindex(x_raw.index, fill_value=0)
        else:
            a = pd.DataFrame(0, index=x_raw.index, columns=TREATMENT_FEATURES)
        a = a.reindex(columns=TREATMENT_FEATURES, fill_value=0)

        # --- Raw values for visualization ---
        x_raw_filled = x_raw.fillna(0.0)
        x_raw_tensor = torch.tensor(x_raw_filled.values, dtype=torch.float32)

        # --- Convert to tensors ---
        x_tensor = torch.tensor(x_norm_filled.values, dtype=torch.float32)
        mask_tensor = torch.tensor(obs_mask.values, dtype=torch.float32)
        dt_tensor = torch.tensor(delta_t.values, dtype=torch.float32).clamp(0, 999)
        a_tensor = torch.tensor(a.values, dtype=torch.float32)

        # --- Extract full-sequence windows ---
        T = len(x_tensor)
        for start in range(0, T - TOTAL_STEPS + 1, STRIDE_STEPS):
            end = start + TOTAL_STEPS
            ctx_end = start + CONTEXT_STEPS

            # Target mask: observations in target region only
            tgt_mask = mask_tensor[ctx_end:end]
            if tgt_mask.sum() < 3:  # Need at least 3 observations
                continue

            window = {
                'x_full': x_tensor[start:end],         # [48, 15]
                'mask_full': mask_tensor[start:end],    # [48, 15]
                'dt_full': dt_tensor[start:end],        # [48, 15]
                'a_full': a_tensor[start:end],          # [48, 5]
                'target_mask': tgt_mask,                # [12, 15]
                'x_raw_full': x_raw_tensor[start:end],  # [48, 15]
                'subject_id': sub,
                'norm_params': {'mean': x_mean.values, 'std': x_std.values},
            }

            if not torch.isnan(window['x_full']).any():
                all_windows.append(window)

        patients_processed += 1
        if patients_processed % 50 == 0:
            print(f"   ... {patients_processed} patients → {len(all_windows)} windows")

    # Stats
    if all_windows:
        avg_tgt_obs = np.mean([w['target_mask'].sum().item() / (HORIZON_STEPS * INPUT_DIM) * 100
                               for w in all_windows])
    else:
        avg_tgt_obs = 0.0

    print(f"\n   ✓ Processed: {patients_processed} patients")
    print(f"   ✓ Skipped: {patients_skipped}")
    print(f"   ✓ Total windows: {len(all_windows):,}")
    print(f"   ✓ Avg target observation rate: {avg_tgt_obs:.1f}%")
    print(f"   ✓ Window: [{TOTAL_STEPS}, {INPUT_DIM}] per sequence")
    return all_windows


# ============================================================
# 6. Loss Functions
# ============================================================

def masked_loss_dense_sparse(pred, target, target_mask, alpha=1.0):
    """
    Masked MSE: only observed targets contribute.
    L = L_vitals + α·L_labs
    """
    sq_err = (pred - target) ** 2

    # Dense vitals (0-3)
    d_mask = target_mask[:, :, DENSE_INDICES]
    d_err = sq_err[:, :, DENSE_INDICES]
    L_vitals = (d_err * d_mask).sum() / (d_mask.sum() + 1e-8)

    # Sparse labs (4-14)
    s_mask = target_mask[:, :, SPARSE_INDICES]
    s_err = sq_err[:, :, SPARSE_INDICES]
    L_labs = (s_err * s_mask).sum() / (s_mask.sum() + 1e-8)

    return L_vitals + alpha * L_labs, L_vitals.item(), L_labs.item()


def hawkes_nll(intensities, events):
    """
    Poisson NLL for treatment intensity.
    L = λ - k·log(λ + ε)
    """
    return (intensities - events * torch.log(intensities + 1e-6)).mean()


# ============================================================
# 7. Training Loop
# ============================================================

def train_v5(train_windows, test_windows, device,
             epochs=30, batch_size=32, lr=1e-3,
             alpha=1.0, lambda_cdsp=0.1, lambda_hawkes=0.01,
             cdsp_warmup=3):
    """
    Train with combined loss:
      L = L_recon + λ_cdsp·L_cdsp + λ_hawkes·L_hawkes

    Teacher-forced: full sequence through Mamba, loss on target positions.
    h[t] predicts x[t+1].
    """
    hidden_dim = 64
    model = DigitalTwinSSM(INPUT_DIM, TREAT_DIM, hidden_dim, INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_loader = DataLoader(ICUWindowDataset(train_windows), batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(ICUWindowDataset(test_windows), batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)

    print(f"\n{'='*70}")
    print("DIGITAL TWIN v5 — TRAINING")
    print(f"{'='*70}")
    print(f"  Model: DigitalTwinSSM (Mamba d_model={hidden_dim}, d_state=16, d_conv=4)")
    print(f"  Input: {INPUT_DIM}*3 + {TREAT_DIM} = {INPUT_DIM*3+TREAT_DIM} dims/step")
    print(f"  Sequence: {TOTAL_STEPS} steps ({CONTEXT_STEPS} ctx + {HORIZON_STEPS} tgt)")
    print(f"  Loss: L_recon + {lambda_cdsp}·L_cdsp + {lambda_hawkes}·L_hawkes")
    print(f"  CDSP warmup: {cdsp_warmup} epochs")
    print(f"  Batch: {batch_size}, LR: {lr}, Epochs: {epochs}")
    print(f"  Train: {len(train_windows):,} windows, Test: {len(test_windows):,}")

    history = {'train': [], 'test': [], 'vital': [], 'lab': [], 'cdsp': [], 'hawkes': []}

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        model.to(device)
        e_loss = e_vit = e_lab = e_cdsp = e_hawk = 0.0
        n_batch = 0

        cdsp_w = 0.0 if epoch < cdsp_warmup else lambda_cdsp

        for batch in train_loader:
            x = batch['x_full'].to(device)       # [B, 48, 15]
            mask = batch['mask_full'].to(device)
            dt = batch['dt_full'].to(device)
            a = batch['a_full'].to(device)
            tm = batch['target_mask'].to(device)  # [B, 12, 15]

            # Forward (full sequence, teacher-forced)
            y_pred, y_var, hawkes_int, cdsp_loss, h_seq = model(x, mask, dt, a)

            # h[t] predicts x[t+1]:
            # predictions for target window = y_pred[:, ctx-1 : ctx+horizon-1]
            pred_target = y_pred[:, CONTEXT_STEPS-1 : CONTEXT_STEPS+HORIZON_STEPS-1]
            true_target = x[:, CONTEXT_STEPS : CONTEXT_STEPS+HORIZON_STEPS]

            # 1. Reconstruction loss (masked)
            loss_recon, l_vit, l_lab = masked_loss_dense_sparse(
                pred_target, true_target, tm, alpha)

            # 2. CDSP decorrelation
            loss_cdsp = cdsp_w * cdsp_loss

            # 3. Hawkes NLL
            loss_hawkes = lambda_hawkes * hawkes_nll(hawkes_int, a)

            # Total
            total_loss = loss_recon + loss_cdsp + loss_hawkes

            if torch.isnan(total_loss):
                print(f"  ⚠ NaN loss at epoch {epoch+1}, skipping batch")
                continue

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient check (first epoch only)
            if epoch == 0 and n_batch == 0:
                has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                             for p in model.parameters())
                print(f"  ✓ Gradient NaN check: {'⚠ NaN detected!' if has_nan else 'PASS'}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            e_loss += loss_recon.item()
            e_vit += l_vit
            e_lab += l_lab
            e_cdsp += cdsp_loss.item() if torch.is_tensor(cdsp_loss) else cdsp_loss
            e_hawk += loss_hawkes.item()
            n_batch += 1

        scheduler.step()
        nb = max(n_batch, 1)
        history['train'].append(e_loss / nb)
        history['vital'].append(e_vit / nb)
        history['lab'].append(e_lab / nb)
        history['cdsp'].append(e_cdsp / nb)
        history['hawkes'].append(e_hawk / nb)

        # --- TEST ---
        model.eval()
        t_loss = 0.0
        t_batch = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x_full'].to(device)
                mask = batch['mask_full'].to(device)
                dt = batch['dt_full'].to(device)
                a = batch['a_full'].to(device)
                tm = batch['target_mask'].to(device)

                y_pred, _, _, _, _ = model(x, mask, dt, a)
                pred_target = y_pred[:, CONTEXT_STEPS-1 : CONTEXT_STEPS+HORIZON_STEPS-1]
                true_target = x[:, CONTEXT_STEPS : CONTEXT_STEPS+HORIZON_STEPS]

                loss, _, _ = masked_loss_dense_sparse(pred_target, true_target, tm, alpha)
                t_loss += loss.item()
                t_batch += 1

        history['test'].append(t_loss / max(t_batch, 1))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  E{epoch+1:3d}/{epochs} | Recon: {history['train'][-1]:.4f} "
                  f"(V:{e_vit/nb:.4f} L:{e_lab/nb:.4f}) "
                  f"CDSP:{e_cdsp/nb:.5f} Hawk:{e_hawk/nb:.4f} "
                  f"| Test: {history['test'][-1]:.4f} "
                  f"| LR: {optimizer.param_groups[0]['lr']:.1e}")

    # --- Post-training metrics ---
    model.cpu().eval()
    metrics = compute_metrics_v5(model, test_loader, test_windows)

    return {
        'model': model, 'history': history,
        'patient_metrics': metrics,
        'final_train': history['train'][-1],
        'final_test': history['test'][-1],
    }


# ============================================================
# 8. Metrics (masked, per-feature)
# ============================================================

def compute_metrics_v5(model, test_loader, test_windows):
    """Compute masked metrics + rollout example."""
    model.eval()
    all_metrics = []
    win_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x_full']
            mask = batch['mask_full']
            dt = batch['dt_full']
            a = batch['a_full']
            tm = batch['target_mask']

            # Teacher-forced predictions
            y_pred, y_var, _, _, _ = model(x, mask, dt, a)
            pred_tgt = y_pred[:, CONTEXT_STEPS-1:CONTEXT_STEPS+HORIZON_STEPS-1]
            true_tgt = x[:, CONTEXT_STEPS:CONTEXT_STEPS+HORIZON_STEPS]

            B = x.shape[0]
            for i in range(B):
                pi = pred_tgt[i].numpy()
                ti = true_tgt[i].numpy()
                tmi = tm[i].numpy()

                # Overall masked metrics
                obs = tmi > 0.5
                if obs.sum() < 3:
                    win_idx += 1
                    continue

                po, to_ = pi[obs], ti[obs]
                mse = np.mean((po - to_) ** 2)
                mae = np.mean(np.abs(po - to_))
                rmse = np.sqrt(mse)
                try:
                    r2 = r2_score(to_, po)
                except:
                    r2 = 0.0
                try:
                    pearson = stats.pearsonr(to_, po)[0]
                    if np.isnan(pearson): pearson = 0.0
                except:
                    pearson = 0.0

                # Per-feature
                pf = {}
                for f in range(INPUT_DIM):
                    fm = tmi[:, f] > 0.5
                    if fm.sum() < 2:
                        pf[f'mse_f{f}'] = pf[f'mae_f{f}'] = pf[f'r2_f{f}'] = np.nan
                        continue
                    pf[f'mse_f{f}'] = np.mean((pi[fm, f] - ti[fm, f]) ** 2)
                    pf[f'mae_f{f}'] = np.mean(np.abs(pi[fm, f] - ti[fm, f]))
                    try:
                        pf[f'r2_f{f}'] = r2_score(ti[fm, f], pi[fm, f])
                    except:
                        pf[f'r2_f{f}'] = 0.0

                # Raw values for viz
                w = test_windows[win_idx] if win_idx < len(test_windows) else None
                all_metrics.append({
                    'mse': mse, 'mae': mae, 'rmse': rmse,
                    'r2': r2, 'pearson': pearson,
                    'predictions': pi, 'ground_truth': ti,
                    'target_mask': tmi,
                    'norm_params': w['norm_params'] if w else None,
                    'x_raw_full': w['x_raw_full'].numpy() if w else None,
                    **pf,
                })
                win_idx += 1

    # Rollout example (first test window)
    if test_windows:
        w0 = test_windows[0]
        x_ctx = w0['x_full'][:CONTEXT_STEPS].unsqueeze(0)
        m_ctx = w0['mask_full'][:CONTEXT_STEPS].unsqueeze(0)
        d_ctx = w0['dt_full'][:CONTEXT_STEPS].unsqueeze(0)
        a_ctx = w0['a_full'][:CONTEXT_STEPS].unsqueeze(0)
        a_fut = w0['a_full'][CONTEXT_STEPS:].unsqueeze(0)

        rollout_pred, rollout_var = model.rollout(x_ctx, m_ctx, d_ctx, a_ctx, a_fut, HORIZON_STEPS)
        print(f"\n  ✓ Rollout example: context [{x_ctx.shape}] → prediction [{rollout_pred.shape}]")
        print(f"    Pred mean range: [{rollout_pred.min():.3f}, {rollout_pred.max():.3f}]")
        print(f"    Pred var range:  [{rollout_var.min():.4f}, {rollout_var.max():.4f}]")

    return all_metrics


# ============================================================
# 9. Visualization
# ============================================================

def create_publication_viz(results, output_file='digital_twin_v5_results.png'):
    """20-panel comprehensive visualization."""
    pm = results['patient_metrics']
    mdf = pd.DataFrame([{k: v for k, v in p.items() if not isinstance(v, np.ndarray)}
                         for p in pm])
    h = results['history']

    fig = plt.figure(figsize=(28, 20))
    fig.suptitle('Digital Twin v5 — Full-Sequence SSM + CDSP + Hawkes',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. Learning curves
    ax = plt.subplot(4, 5, 1)
    ax.plot(h['train'], label='Train', lw=2.5, color='#2E86AB')
    ax.plot(h['test'], label='Test', lw=2.5, ls='--', color='#A23B72')
    ax.set_title('Learning Curves', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Vital vs Lab loss
    ax = plt.subplot(4, 5, 2)
    ax.plot(h['vital'], label='Vitals', lw=2, color='#06A77D')
    ax.plot(h['lab'], label='Labs', lw=2, color='#F18F01')
    ax.set_title('Dense vs Sparse Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

    # 3. CDSP + Hawkes losses
    ax = plt.subplot(4, 5, 3)
    ax.plot(h['cdsp'], label='CDSP', lw=2, color='#E63946')
    ax.plot(h['hawkes'], label='Hawkes', lw=2, color='#457B9D')
    ax.set_title('Regularization Losses', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

    # 4. MSE Distribution
    ax = plt.subplot(4, 5, 4)
    ax.hist(mdf['mse'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax.axvline(mdf['mse'].median(), color='red', ls='--',
               label=f"Med: {mdf['mse'].median():.4f}")
    ax.set_title('MSE Distribution', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 5. Summary Table
    ax = plt.subplot(4, 5, 5)
    ax.axis('off')
    tbl = [['Metric', 'Value'],
           ['MSE', f"{mdf['mse'].mean():.4f}"],
           ['MAE', f"{mdf['mae'].mean():.4f}"],
           ['RMSE', f"{mdf['rmse'].mean():.4f}"],
           ['R²', f"{mdf['r2'].mean():.4f}"],
           ['Pearson', f"{mdf['pearson'].mean():.3f}"],
           ['Windows', f"{len(mdf)}"]]
    t = ax.table(cellText=tbl, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 2)
    for j in range(2):
        t[(0, j)].set_facecolor('#06A77D')
        t[(0, j)].set_text_props(weight='bold', color='white')
    ax.set_title('Summary (masked)', fontsize=12, fontweight='bold')

    # 6-10: Best 5 windows
    best5 = mdf.nsmallest(5, 'mse').index.tolist()
    for i, idx in enumerate(best5):
        ax = plt.subplot(4, 5, 6 + i)
        p = pm[idx]
        time = np.arange(HORIZON_STEPS)
        tmf = p['target_mask'][:, 0]
        obs_t = time[tmf > 0.5]
        if len(obs_t) > 0:
            ax.plot(obs_t, p['ground_truth'][tmf > 0.5, 0], 'o-', label='True',
                    ms=4, lw=1.5, color='#2E86AB')
        ax.plot(time, p['predictions'][:, 0], '--', label='Pred', lw=1.5, color='#F18F01')
        ax.set_title(f'Best #{i+1} (MSE={p["mse"]:.4f})', fontsize=10,
                    fontweight='bold', color='green')
        if i == 0: ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 11-15: Worst 5
    worst5 = mdf.nlargest(5, 'mse').index.tolist()
    for i, idx in enumerate(worst5):
        ax = plt.subplot(4, 5, 11 + i)
        p = pm[idx]
        time = np.arange(HORIZON_STEPS)
        tmf = p['target_mask'][:, 0]
        obs_t = time[tmf > 0.5]
        if len(obs_t) > 0:
            ax.plot(obs_t, p['ground_truth'][tmf > 0.5, 0], 'o-', label='True',
                    ms=4, lw=1.5, color='#2E86AB')
        ax.plot(time, p['predictions'][:, 0], '--', lw=1.5, color='#A23B72')
        ax.set_title(f'Worst #{i+1} (MSE={p["mse"]:.4f})', fontsize=10,
                    fontweight='bold', color='red')
        ax.grid(True, alpha=0.3)

    # 16-18: Per-feature metrics
    colors = ['#2E86AB' if f in DENSE_INDICES else '#F18F01' for f in range(INPUT_DIM)]
    for panel, metric, title in [(16, 'mse', 'Per-Feature MSE'),
                                  (17, 'r2', 'Per-Feature R²'),
                                  (18, 'mae', 'Per-Feature MAE')]:
        ax = plt.subplot(4, 5, panel)
        vals = [mdf[f'{metric}_f{f}'].mean() for f in range(INPUT_DIM)]
        ax.barh(range(INPUT_DIM), vals, color=colors, alpha=0.8)
        ax.set_yticks(range(INPUT_DIM))
        ax.set_yticklabels(FEATURE_NAMES, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    # 19. Pearson
    ax = plt.subplot(4, 5, 19)
    ax.hist(mdf['pearson'].dropna(), bins=20, color='#06A77D', alpha=0.7, edgecolor='black')
    ax.axvline(mdf['pearson'].mean(), color='red', ls='--',
               label=f"Mean: {mdf['pearson'].mean():.3f}")
    ax.set_title('Pearson Correlation', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 20. Architecture info
    ax = plt.subplot(4, 5, 20)
    ax.axis('off')
    info = (f"Digital Twin v5\n{'─'*28}\n"
            f"SSM: Mamba (full sequence)\n"
            f"CDSP: Orthogonal decorrelation\n"
            f"Hawkes: Latent intensity\n"
            f"Input: x+mask+dt+a ({INPUT_DIM*3+TREAT_DIM}d)\n"
            f"Bins: 2h, No LOV/imputation\n"
            f"Loss: masked (dense+sparse)\n"
            f"Train: {results['final_train']:.4f}\n"
            f"Test: {results['final_test']:.4f}")
    ax.text(0.5, 0.5, info, transform=ax.transAxes, fontsize=10,
            va='center', ha='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def create_per_feature_report(results, output_file='digital_twin_v5_features.png'):
    """Per-feature table and R² chart."""
    pm = results['patient_metrics']
    mdf = pd.DataFrame([{k: v for k, v in p.items() if not isinstance(v, np.ndarray)}
                         for p in pm])

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
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
                     f'{np.sqrt(m):.4f}', f'{r:.4f}'])
    cell = [header] + rows
    t = axes[0].table(cellText=cell, cellLoc='center', loc='center',
                      colWidths=[0.2, 0.1, 0.15, 0.15, 0.15, 0.15])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 2.0)
    for j in range(6):
        t[(0, j)].set_facecolor('#2E86AB')
        t[(0, j)].set_text_props(weight='bold', color='white')
    for i in range(1, 16):
        color = '#E8F4F8' if (i-1) in DENSE_INDICES else '#FFF3E0'
        for j in range(6):
            t[(i, j)].set_facecolor(color)

    # R² bar chart
    feat_r2 = [mdf[f'r2_f{f}'].mean() for f in range(INPUT_DIM)]
    colors = ['#2E86AB' if f in DENSE_INDICES else '#F18F01' for f in range(INPUT_DIM)]
    axes[1].barh(range(INPUT_DIM), feat_r2, color=colors, alpha=0.85)
    axes[1].set_yticks(range(INPUT_DIM))
    axes[1].set_yticklabels(FEATURE_NAMES, fontsize=10)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('R²')
    axes[1].set_title('R² per Feature (blue=dense, orange=sparse)', fontweight='bold')
    axes[1].axvline(0, color='black', lw=0.5)
    axes[1].grid(True, alpha=0.3, axis='x')
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color='#2E86AB', label='Dense'),
                            Patch(color='#F18F01', label='Sparse')], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()
    return mdf


def counterfactual_demo(model, test_windows, output_file='digital_twin_v5_counterfactual.png'):
    """Counterfactual simulation: toggle treatment, compare trajectories."""
    if not test_windows:
        return

    w = test_windows[0]
    x_ctx = w['x_full'][:CONTEXT_STEPS].unsqueeze(0)
    m_ctx = w['mask_full'][:CONTEXT_STEPS].unsqueeze(0)
    d_ctx = w['dt_full'][:CONTEXT_STEPS].unsqueeze(0)
    a_ctx = w['a_full'][:CONTEXT_STEPS].unsqueeze(0)
    a_fut = w['a_full'][CONTEXT_STEPS:].unsqueeze(0)

    # Factual
    pred_f, var_f = model.rollout(x_ctx, m_ctx, d_ctx, a_ctx, a_fut, HORIZON_STEPS)

    # Counterfactual: toggle first treatment
    a_cf = a_fut.clone()
    a_cf[:, :, 0] = 1.0 - a_cf[:, :, 0]
    pred_cf, var_cf = model.rollout(x_ctx, m_ctx, d_ctx, a_ctx, a_cf, HORIZON_STEPS)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    time = np.arange(HORIZON_STEPS) * 2  # Convert to hours

    for i in range(min(6, INPUT_DIM)):
        ax = axes[i // 3, i % 3]
        f_mean = pred_f[0, :, i].detach().numpy()
        f_std = torch.sqrt(var_f[0, :, i]).detach().numpy()
        cf_mean = pred_cf[0, :, i].detach().numpy()

        ax.plot(time, f_mean, label='Factual', lw=2.5, color='#2E86AB')
        ax.fill_between(time, f_mean - 2*f_std, f_mean + 2*f_std,
                        alpha=0.2, color='#2E86AB')
        ax.plot(time, cf_mean, '--', label='Counterfactual', lw=2.5, color='#A23B72')
        ax.set_title(FEATURE_NAMES[i], fontsize=11, fontweight='bold')
        ax.set_xlabel('Hours'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Counterfactual: Toggle {TREATMENT_NAMES[0]}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# ============================================================
# 10. Main
# ============================================================

def main():
    # 2% of dataset ≈ 1307 patients
    all_windows = load_data_v5(target_subjects=1307)

    if not all_windows:
        print("ERROR: No windows. Check data paths.")
        return

    random.seed(42)
    random.shuffle(all_windows)
    split = int(len(all_windows) * 0.8)
    train_w, test_w = all_windows[:split], all_windows[split:]

    print(f"\n📊 Dataset: {len(all_windows):,} total, {len(train_w):,} train, {len(test_w):,} test")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Device: {device}")

    results = train_v5(train_w, test_w, device, epochs=30, batch_size=32,
                       lambda_cdsp=0.1, lambda_hawkes=0.01, cdsp_warmup=3)

    # Save
    torch.save(results['model'].state_dict(), 'digital_twin_v5.pt')
    print(f"   ✓ Saved: digital_twin_v5.pt")

    mdf = pd.DataFrame([{k: v for k, v in p.items() if not isinstance(v, np.ndarray)}
                         for p in results['patient_metrics']])
    mdf.to_csv('metrics_v5.csv', index=False)
    print(f"   ✓ Saved: metrics_v5.csv ({len(mdf)} windows)")

    # Visualizations
    print("\n📊 Generating visualizations...")
    create_publication_viz(results)
    create_per_feature_report(results)
    counterfactual_demo(results['model'], test_w)

    # Final summary
    print(f"\n{'='*70}")
    print("DIGITAL TWIN v5 — RESULTS")
    print(f"{'='*70}")
    print(f"  Train Loss: {results['final_train']:.6f}")
    print(f"  Test Loss:  {results['final_test']:.6f}")
    print(f"\n  Metrics (masked, observed only):")
    print(f"    MSE:     {mdf['mse'].mean():.4f}")
    print(f"    MAE:     {mdf['mae'].mean():.4f}")
    print(f"    RMSE:    {mdf['rmse'].mean():.4f}")
    print(f"    R²:      {mdf['r2'].mean():.4f}")
    print(f"    Pearson: {mdf['pearson'].mean():.4f}")

    print(f"\n  Per-Feature:")
    print(f"   {'Feature':<14} {'MSE':>8} {'MAE':>8} {'R²':>10} {'Type'}")
    for f in range(INPUT_DIM):
        m = mdf[f'mse_f{f}'].mean()
        a = mdf[f'mae_f{f}'].mean()
        r = mdf[f'r2_f{f}'].mean()
        ft = 'D' if f in DENSE_INDICES else 'S'
        print(f"   {FEATURE_NAMES[f]:<14} {m:8.4f} {a:8.4f} {r:10.4f} [{ft}]")

    print(f"\n📄 Files: digital_twin_v5_results.png, digital_twin_v5_features.png, "
          f"digital_twin_v5_counterfactual.png, digital_twin_v5.pt, metrics_v5.csv")


if __name__ == "__main__":
    main()
