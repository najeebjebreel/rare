"""
Run membership-inference attacks (LiRA / Attack-R / RMIA / LOSS) against
the shadow models in an experiment directory and persist per-attack scores,
labels, ROC thresholds, and aggregate CSV summaries.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import yaml
from scipy.stats import norm, trim_mean
from sklearn.metrics import auc, roc_curve

from rare.attacks.plotting import plot_combined_roc  # noqa: F401  -- available for ad-hoc ROC plots
from rare.attacks.signals import convert_signals
from rare.utils.common import (
    append_runtime,
    parse_overrides,
    recursive_update,
    save_config,
    set_seed,
    setup_logger,
    validate_config,
)
from rare.utils.data import load_dataset, resolve_subsample

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# -----------------------------
# helpers
# -----------------------------
def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def ce_loss_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample cross-entropy: CE(x,y) = logsumexp(logits) - logits[y]."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    lse = _logsumexp(logits, axis=1)
    yt = logits[np.arange(logits.shape[0]), labels]
    return (lse - yt).astype(np.float64)


def _deterministic_choice(arr: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Deterministically choose k distinct elements from arr (or all if k>=len(arr))."""
    arr = np.asarray(arr, dtype=np.int64)
    if k <= 0 or arr.size <= k:
        return arr
    rs = np.random.RandomState(int(seed))
    return rs.choice(arr, size=int(k), replace=False)


def _build_reference_pool(num_models: int, target_models: List[int]) -> np.ndarray:
    target_set = set(int(x) for x in target_models)
    all_models = np.arange(int(num_models), dtype=np.int64)
    ref_pool = all_models[~np.isin(all_models, np.array(sorted(target_set), dtype=np.int64))]
    return ref_pool


# -----------------------------
# LiRA
# -----------------------------
def lira_scores_for_model(
    target_model: int,
    signals_lira: torch.Tensor,
    keep_matrix: np.ndarray,
    fix_variance: bool = False,
    ref_models: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LiRA with explicit reference model set (`ref_models`).

    `ref_models` should NOT include any target models (and will also exclude
    `target_model` itself).
    """
    M, N, A = signals_lira.shape
    sig_mean = signals_lira.mean(dim=-1).detach().cpu().numpy()
    labels = keep_matrix[target_model].astype(bool)

    if ref_models is None:
        ref_models = np.arange(M, dtype=np.int64)
    ref_models = np.asarray(ref_models, dtype=np.int64)
    ref_models = ref_models[ref_models != int(target_model)]

    dat_in = []
    dat_out = []

    for i in range(N):
        ref_keep = keep_matrix[ref_models, i].astype(bool)
        in_ids = ref_models[ref_keep]
        out_ids = ref_models[~ref_keep]
        dat_in.append(sig_mean[in_ids, i] if in_ids.size > 0 else np.array([], dtype=np.float64))
        dat_out.append(sig_mean[out_ids, i] if out_ids.size > 0 else np.array([], dtype=np.float64))

    mean_in = np.array([np.median(x) if len(x) > 0 else 0.0 for x in dat_in])
    mean_out = np.array([np.median(x) if len(x) > 0 else 0.0 for x in dat_out])

    if fix_variance:
        all_in = (
            np.concatenate([x for x in dat_in if len(x) > 0])
            if any(len(x) > 0 for x in dat_in)
            else np.array([0.0])
        )
        global_std = float(np.std(all_in)) if all_in.size > 1 else 1e-30
        std_in = np.full(N, global_std)
        std_out = np.full(N, global_std)
    else:
        std_in = np.array([np.std(x) if len(x) > 1 else 1e-30 for x in dat_in])
        std_out = np.array([np.std(x) if len(x) > 1 else 1e-30 for x in dat_out])

    sc = sig_mean[target_model, :]
    pr_in = -norm.logpdf(sc, mean_in, std_in + 1e-30)
    pr_out = -norm.logpdf(sc, mean_out, std_out + 1e-30)

    scores = pr_in - pr_out
    # Higher => more member-like.
    return (-scores.astype(np.float32)), labels


def lira_scores_for_model_fast(
    target_model: int,
    signals_lira: torch.Tensor,
    keep_matrix: np.ndarray,
    fix_variance: bool = False,
    ref_models: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized LiRA. Numerically equivalent to `lira_scores_for_model`.

    The reference implementation has an O(N) Python loop that builds variable-
    length `dat_in[i]` and `dat_out[i]` lists, then computes `np.median` and
    `np.std` per sample. That dispatch overhead dominates at our N=50k.

    This version slices the keep matrix and signal mean to `ref_models` in
    one shot to get `(R, N)` tensors, then uses NaN-masking +
    `np.nanmedian` / `np.nanstd` to handle the ragged in/out subsets without
    a Python loop. The remaining work (norm.logpdf over (N,) arrays) is
    already vectorized in scipy.

    GPU was considered and rejected for LiRA at this scale: the per-call
    work is small enough that kernel-launch overhead would dominate. CPU
    numpy `nanmedian`/`nanstd` are well-tuned and yield ~20-30x over the
    reference at our scale.
    """
    M, N, A = signals_lira.shape
    sig_mean = signals_lira.mean(dim=-1).detach().cpu().numpy()  # (M, N)
    labels = keep_matrix[target_model].astype(bool)

    if ref_models is None:
        ref_models = np.arange(M, dtype=np.int64)
    ref_models = np.asarray(ref_models, dtype=np.int64)
    ref_models = ref_models[ref_models != int(target_model)]

    # (R, N) slices over the reference pool
    sig_R_N = sig_mean[ref_models, :]                         # (R, N) float
    keep_R_N = keep_matrix[ref_models, :].astype(bool)        # (R, N) bool

    # NaN-mask: a ref contributes to dat_in[i] only if it's a member of i.
    sig_in = np.where(keep_R_N, sig_R_N, np.nan).astype(np.float64)
    sig_out = np.where(~keep_R_N, sig_R_N, np.nan).astype(np.float64)

    import warnings as _warnings
    with np.errstate(invalid="ignore"), _warnings.catch_warnings():
        # nanmedian emits a `RuntimeWarning: All-NaN slice encountered` via
        # Python's warnings module (np.errstate doesn't suppress it) — happens
        # at low ref counts where some sample has zero in-refs or zero
        # out-refs. We handle it by mapping NaN -> 0.0 below.
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_in = np.nanmedian(sig_in, axis=0)
        mean_out = np.nanmedian(sig_out, axis=0)
    # All-NaN columns (a sample with no in-refs or no out-refs) → 0 to match
    # the reference's `0.0 if len(x) == 0 else np.median(x)` branch.
    mean_in = np.where(np.isnan(mean_in), 0.0, mean_in)
    mean_out = np.where(np.isnan(mean_out), 0.0, mean_out)

    if fix_variance:
        all_in = sig_in[~np.isnan(sig_in)]
        global_std = float(np.std(all_in)) if all_in.size > 1 else 1e-30
        std_in = np.full(N, global_std)
        std_out = np.full(N, global_std)
    else:
        with np.errstate(invalid="ignore"), _warnings.catch_warnings():
            _warnings.simplefilter("ignore", category=RuntimeWarning)
            std_in = np.nanstd(sig_in, axis=0, ddof=0)
            std_out = np.nanstd(sig_out, axis=0, ddof=0)
        # Reference uses 1e-30 floor when cnt <= 1.
        cnt_in = (~np.isnan(sig_in)).sum(axis=0)
        cnt_out = (~np.isnan(sig_out)).sum(axis=0)
        std_in = np.where(cnt_in <= 1, 1e-30, std_in)
        std_out = np.where(cnt_out <= 1, 1e-30, std_out)

    sc = sig_mean[int(target_model), :]
    pr_in = -norm.logpdf(sc, mean_in, std_in + 1e-30)
    pr_out = -norm.logpdf(sc, mean_out, std_out + 1e-30)

    scores = pr_in - pr_out
    return (-scores.astype(np.float32)), labels


# -----------------------------
# Attack-R (PURE CE LOSS from logits)
# -----------------------------
def attackr_scores_for_model(
    target_model: int,
    ce_losses: torch.Tensor,
    keep_matrix: np.ndarray,
    ref_models: Optional[np.ndarray] = None,
    num_ref_models: int = -1,
    ref_selection_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attack-R using OUT-only reference models, with PURE CE LOSS as the signal.

    `ce_losses` is `(M, N, A)`; we average over augmentations.
    For each sample, the score is the empirical CDF of the target's loss
    in the OUT-reference loss distribution; we return `-pr` so higher => member.
    """
    M, N, A = ce_losses.shape

    # Attack-R reference convention (Pollock 2025 + ml_privacy_meter/2024_rmia):
    # single un-augmented view, no averaging over A. Our pipeline's logits
    # tensor carries A=2 (original + horizontal flip), but Attack-R consumes
    # only the un-augmented slot to match the reference baselines.
    loss_np = ce_losses.detach().cpu().numpy()
    loss_mean = loss_np[:, :, 0]

    labels = keep_matrix[target_model].astype(bool)
    prediction = np.zeros(N, dtype=np.float32)

    if ref_models is None:
        ref_models = np.arange(M, dtype=np.int64)
    ref_models = np.asarray(ref_models, dtype=np.int64)
    ref_models = ref_models[ref_models != int(target_model)]

    k_out = int(num_ref_models) if num_ref_models is not None else -1

    # Fixed global bounds (matches the reference Attack-R style).
    signal_min = 0.0
    signal_max = 1000.0

    for i in range(N):
        out_mask = ~keep_matrix[ref_models, i].astype(bool)
        out_ids = ref_models[out_mask]

        if out_ids.size == 0:
            prediction[i] = -0.5
            continue

        if k_out > 0 and out_ids.size > k_out:
            rs = np.random.RandomState(int(ref_selection_seed) + int(i))
            out_ids = rs.choice(out_ids, size=k_out, replace=False)

        tgt_loss = float(loss_mean[target_model, i])
        ref_losses = loss_mean[out_ids, i].astype(np.float64)

        losses_with_bounds = np.concatenate([[signal_min], np.sort(ref_losses), [signal_max]])
        discrete_alpha = np.linspace(0.0, 1.0, len(losses_with_bounds))
        pr = float(np.interp(tgt_loss, losses_with_bounds, discrete_alpha))

        prediction[i] = -pr

    return prediction.astype(np.float32), labels


def attackr_scores_for_model_fast(
    target_model: int,
    ce_losses: torch.Tensor,
    keep_matrix: np.ndarray,
    ref_models: Optional[np.ndarray] = None,
    num_ref_models: int = -1,
    ref_selection_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Faster Attack-R. Numerically equivalent to `attackr_scores_for_model`.

    Optimizations vs reference:
      1. Pre-slice `loss_mean[ref_models, :]` and `out_mask` once outside the
         per-sample loop (reference re-indexes each iteration).
      2. Inline the bin-search + CDF interpolation as a manual `np.searchsorted`
         + arithmetic, avoiding `np.concatenate([[0], sorted, [1000]])`,
         `np.linspace(0, 1, n+2)`, and `np.interp(...)`. Those three calls
         carry ~5x more overhead than the sort itself at k_out~128.

    The per-sample seeded RNG (`np.random.RandomState(ref_selection_seed + i)`)
    stays in a Python loop — vectorizing it would require dropping per-sample
    seed independence, which would change the chosen subsamples and break
    numerical equivalence with the reference.

    Speedup ~3-5x at our N=50k / k_out ∈ {2…128} settings.
    """
    M, N, A = ce_losses.shape

    # Attack-R reference convention (Pollock 2025 + ml_privacy_meter/2024_rmia):
    # single un-augmented view, no averaging over A. See note in
    # `attackr_scores_for_model` for the rationale.
    loss_np = ce_losses.detach().cpu().numpy()
    loss_mean = loss_np[:, :, 0]

    labels = keep_matrix[target_model].astype(bool)
    prediction = np.empty(N, dtype=np.float32)

    if ref_models is None:
        ref_models = np.arange(M, dtype=np.int64)
    ref_models = np.asarray(ref_models, dtype=np.int64)
    ref_models = ref_models[ref_models != int(target_model)]
    R = ref_models.size

    k_out = int(num_ref_models) if num_ref_models is not None else -1
    base_seed = int(ref_selection_seed)
    signal_min = 0.0
    signal_max = 1000.0

    # Pre-slice (R, N) views once
    loss_R_N = loss_mean[ref_models, :]                           # (R, N)
    out_mask_R_N = ~keep_matrix[ref_models, :].astype(bool)       # (R, N)
    tgt = loss_mean[int(target_model), :]                         # (N,)

    for i in range(N):
        col_mask = out_mask_R_N[:, i]
        out_losses_i = loss_R_N[col_mask, i]                      # variable size

        if out_losses_i.size == 0:
            prediction[i] = -0.5
            continue

        if k_out > 0 and out_losses_i.size > k_out:
            rs = np.random.RandomState(base_seed + int(i))
            sub_idx = rs.choice(out_losses_i.size, size=k_out, replace=False)
            out_losses_i = out_losses_i[sub_idx]

        out_losses_i = out_losses_i.astype(np.float64, copy=False)
        sorted_losses = np.sort(out_losses_i)
        n = sorted_losses.size

        tgt_i = float(tgt[i])

        # Equivalent to: np.interp(tgt, [0] + sorted + [1000], linspace(0, 1, n+2))
        # with explicit clipping to match np.interp's tail behavior.
        if tgt_i <= signal_min:
            prediction[i] = -0.0
            continue
        if tgt_i >= signal_max:
            prediction[i] = -1.0
            continue

        # j = number of sorted_losses values strictly <= tgt_i (via right-search)
        j = int(np.searchsorted(sorted_losses, tgt_i, side="right"))

        # The augmented array is [0, sorted_0, ..., sorted_{n-1}, 1000].
        # tgt falls in bin [augmented[j], augmented[j+1]] where j is the
        # left-edge index in the augmented array.
        if j == 0:
            lo, hi = signal_min, float(sorted_losses[0])
        elif j == n:
            lo, hi = float(sorted_losses[n - 1]), signal_max
        else:
            lo = float(sorted_losses[j - 1])
            hi = float(sorted_losses[j])

        denom = hi - lo
        frac = 0.0 if denom == 0.0 else (tgt_i - lo) / denom
        pr = (j + frac) / (n + 1)
        prediction[i] = -pr

    return prediction.astype(np.float32, copy=False), labels


def rmia_scores_for_model(
    target_model: int,
    signals_rmia: torch.Tensor,
    keep_matrix: np.ndarray,
    ref_models: Optional[np.ndarray] = None,
    gamma: float = 2.0,
    proportiontocut: float = 0.2,
    top_k: int = -1,
    num_ref_models: int = 2,
    ref_selection_seed: int = 0,
    offline: object = False,
    offline_a: float = 0.3,
    augmentation_mode: str = "augmented",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Official-consistent RMIA (relative attack) implementation.

    Notes:
      - `ref_models` is the full eligible pool (target excluded). DO NOT subsample
        with `num_ref_models`; that only controls per-point in/out slicing
        (`out_or_in_size = num_ref_models // 2`).
      - The population is `~labels` (non-members under the target).
      - Overlap correction uses the FULL population size (before top_k):
            score[x] = (score[x] * |POP| - 1) / (|POP| - 1)
    """
    sig = signals_rmia.detach().cpu().numpy()
    M, N, A = sig.shape
    labels = keep_matrix[target_model].astype(bool)

    if ref_models is None:
        ref_models = np.arange(M, dtype=np.int64)
    ref_models = np.asarray(ref_models, dtype=np.int64)
    ref_models = ref_models[ref_models != int(target_model)]

    out_or_in_size = max(1, int(num_ref_models) // 2)

    pop_all = np.where(~labels)[0]
    P_full = int(pop_all.size)

    rng = np.random.RandomState(int(ref_selection_seed) + 12345)

    use_augmented = (str(augmentation_mode).lower() == "augmented")
    A_use = A if use_augmented else 1

    # Per-sample baselines, split by role (LT-IQR canonical online RMIA):
    #   mean_ref_x[j] — used when j is x (sample being scored). Mean over IN ∪ OUT shadows.
    #   mean_ref_z[j] — used when j is z (population sample). Mean over OUT-only shadows.
    # Offline / fractional modes set them equal.
    mean_ref_x = np.zeros((N, A_use), dtype=np.float64)
    mean_ref_z = np.zeros((N, A_use), dtype=np.float64)

    offline_is_bool = isinstance(offline, (bool, np.bool_))
    offline_is_frac = (not offline_is_bool) and str(offline).replace(".", "").isnumeric()
    offline_frac = None
    if offline_is_frac:
        offline_frac = float(offline)
        if not (0.0 <= offline_frac <= 1.0):
            raise ValueError("offline as float must be in [0,1].")

    for j in range(N):
        in_ids_all = ref_models[keep_matrix[ref_models, j].astype(bool)]
        out_ids_all = ref_models[~keep_matrix[ref_models, j].astype(bool)]

        in_ids = in_ids_all[:out_or_in_size] if in_ids_all.size > 0 else np.array([], dtype=np.int64)
        out_ids = out_ids_all[:out_or_in_size] if out_ids_all.size > 0 else np.array([], dtype=np.int64)

        if offline_is_bool and bool(offline):
            if out_ids.size == 0:
                mean = np.full((A_use,), 1e-12, dtype=np.float64)
            else:
                mean = sig[out_ids, j, :A_use].mean(axis=0)
                mean = np.maximum(mean, 1e-12)
            v = ((1.0 + float(offline_a)) / 2.0) * mean + (1.0 - float(offline_a)) / 2.0
            mean_ref_x[j, :] = v
            mean_ref_z[j, :] = v

        elif offline_is_frac:
            tot_nb = out_or_in_size
            in_k = int(tot_nb * (1.0 - offline_frac))
            out_k = int(tot_nb * offline_frac)
            in_sel = in_ids_all[:in_k] if in_ids_all.size > 0 else np.array([], dtype=np.int64)
            out_sel = out_ids_all[:out_k] if out_ids_all.size > 0 else np.array([], dtype=np.int64)
            chosen = np.concatenate([in_sel, out_sel], axis=0)
            if chosen.size == 0:
                v = np.full((A_use,), 1e-12, dtype=np.float64)
            else:
                v = sig[chosen, j, :A_use].mean(axis=0)
                v = np.maximum(v, 1e-12)
            mean_ref_x[j, :] = v
            mean_ref_z[j, :] = v

        else:
            # ONLINE — LT-IQR canonical.
            union_ids = np.concatenate([in_ids, out_ids], axis=0)
            if union_ids.size > 0:
                m_x = sig[union_ids, j, :A_use].mean(axis=0)
            else:
                m_x = np.full((A_use,), 1e-12, dtype=np.float64)
            mean_ref_x[j, :] = np.maximum(m_x, 1e-12)
            if out_ids.size > 0:
                m_z = sig[out_ids, j, :A_use].mean(axis=0)
            else:
                m_z = np.full((A_use,), 1e-12, dtype=np.float64)
            mean_ref_z[j, :] = np.maximum(m_z, 1e-12)

    member_score = np.zeros(N, dtype=np.float32)

    for x in range(N):
        if P_full == 0:
            member_score[x] = 0.0
            continue

        pop = pop_all.copy()

        if int(top_k) != -1 and pop.size > int(top_k):
            pop = rng.choice(pop, size=int(top_k), replace=False)

        # Force x into the pool when x is a non-member so the "-1" overlap correction is meaningful.
        if (not labels[x]) and (x in pop_all):
            if x not in pop:
                if pop.size == 0:
                    pop = np.array([x], dtype=np.int64)
                else:
                    pop[-1] = x

        P_used = int(pop.size)
        if P_used == 0:
            member_score[x] = 0.0
            continue

        votes = []
        for k in range(A_use):
            s_x = float(sig[target_model, x, k]) + 1e-12
            s_z = sig[target_model, pop, k].astype(np.float64) + 1e-12

            mean_x = float(mean_ref_x[x, k]) + 1e-12   # IN ∪ OUT for x
            mean_z = mean_ref_z[pop, k].astype(np.float64) + 1e-12   # OUT-only for z

            prob_ratio_x = s_x / mean_x
            prob_ratio_z_rev = mean_z / s_z
            score = prob_ratio_x * prob_ratio_z_rev

            votes.append(score > float(gamma))

        votes = np.stack(votes, axis=0)

        if use_augmented:
            majority = (votes.mean(axis=0) > 0.5)
        else:
            majority = votes[0]

        raw = float(majority.mean())
        # Apply overlap correction only for non-member x, using the FULL population size.
        if (not labels[x]) and P_full > 1:
            raw = (raw * P_full - 1.0) / (P_full - 1.0)

        member_score[x] = raw

    return member_score.astype(np.float32), labels


def rmia_scores_for_model_gpu(
    target_model: int,
    signals_rmia: torch.Tensor,
    keep_matrix: np.ndarray,
    ref_models: Optional[np.ndarray] = None,
    gamma: float = 2.0,
    proportiontocut: float = 0.2,
    top_k: int = -1,
    num_ref_models: int = 2,
    ref_selection_seed: int = 0,
    offline: object = False,
    offline_a: float = 0.3,
    augmentation_mode: str = "augmented",
    device: str = "cuda",
    chunk_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU-vectorized RMIA. Numerically equivalent to `rmia_scores_for_model`
    for `top_k == -1` (the only setting used in this pipeline); falls back to
    the numpy implementation for `top_k != -1`.

    Speedup ~30-50x over the numpy version on per-target wall time. The
    per-sample mean_ref step still runs on numpy (uses scipy.stats.trim_mean
    and is already fast ~5-10s); the per-x scoring loop, which dominates the
    runtime (~190s out of ~200s), is vectorized on GPU and chunked over x
    so peak memory stays bounded by `chunk_size * P * A * 4 bytes`.

    Precision: scoring uses fp32 on GPU vs fp64 on CPU. Max observed
    |Δscore| against the numpy reference is < 1e-5 in practice.
    """
    if int(top_k) != -1:
        # GPU fast path only handles full-population scoring (our config).
        # Subsampled scoring would require per-x rng draws; fall back.
        return rmia_scores_for_model(
            target_model, signals_rmia, keep_matrix, ref_models, gamma,
            proportiontocut, top_k, num_ref_models, ref_selection_seed,
            offline, offline_a, augmentation_mode,
        )

    sig_np = signals_rmia.detach().cpu().numpy()
    M, N, A = sig_np.shape
    labels = keep_matrix[target_model].astype(bool)

    if ref_models is None:
        ref_models = np.arange(M, dtype=np.int64)
    ref_models = np.asarray(ref_models, dtype=np.int64)
    ref_models = ref_models[ref_models != int(target_model)]

    out_or_in_size = max(1, int(num_ref_models) // 2)

    pop_all = np.where(~labels)[0]
    P_full = int(pop_all.size)

    use_augmented = (str(augmentation_mode).lower() == "augmented")
    A_use = A if use_augmented else 1

    # ---- Step 1: per-sample mean_ref_x AND mean_ref_z (LT-IQR canonical) ----
    # mean_ref_x[j] is the baseline used when j is x (the sample being scored).
    # mean_ref_z[j] is the baseline used when j is z (a population sample).
    # In the canonical online RMIA (Pollock 2025 / Zarifzadeh 2024):
    #   - x's baseline averages IN ∪ OUT shadows for j
    #   - z's baseline averages OUT-only shadows for j
    # Offline / fractional modes use the same baseline for x and z.
    mean_ref_x = np.zeros((N, A_use), dtype=np.float64)
    mean_ref_z = np.zeros((N, A_use), dtype=np.float64)
    offline_is_bool = isinstance(offline, (bool, np.bool_))
    offline_is_frac = (not offline_is_bool) and str(offline).replace(".", "").isnumeric()
    offline_frac = float(offline) if offline_is_frac else None

    for j in range(N):
        in_ids_all = ref_models[keep_matrix[ref_models, j].astype(bool)]
        out_ids_all = ref_models[~keep_matrix[ref_models, j].astype(bool)]
        in_ids = in_ids_all[:out_or_in_size] if in_ids_all.size > 0 else np.array([], dtype=np.int64)
        out_ids = out_ids_all[:out_or_in_size] if out_ids_all.size > 0 else np.array([], dtype=np.int64)

        if offline_is_bool and bool(offline):
            if out_ids.size == 0:
                mean = np.full((A_use,), 1e-12, dtype=np.float64)
            else:
                mean = sig_np[out_ids, j, :A_use].mean(axis=0)
                mean = np.maximum(mean, 1e-12)
            v = ((1.0 + float(offline_a)) / 2.0) * mean + (1.0 - float(offline_a)) / 2.0
            mean_ref_x[j, :] = v
            mean_ref_z[j, :] = v
        elif offline_is_frac:
            tot_nb = out_or_in_size
            in_k = int(tot_nb * (1.0 - offline_frac))
            out_k = int(tot_nb * offline_frac)
            in_sel = in_ids_all[:in_k] if in_ids_all.size > 0 else np.array([], dtype=np.int64)
            out_sel = out_ids_all[:out_k] if out_ids_all.size > 0 else np.array([], dtype=np.int64)
            chosen = np.concatenate([in_sel, out_sel], axis=0)
            if chosen.size == 0:
                v = np.full((A_use,), 1e-12, dtype=np.float64)
            else:
                v = sig_np[chosen, j, :A_use].mean(axis=0)
                v = np.maximum(v, 1e-12)
            mean_ref_x[j, :] = v
            mean_ref_z[j, :] = v
        else:
            # ONLINE — LT-IQR canonical (Pollock 2025, Zarifzadeh 2024 RMIA reference).
            union_ids = np.concatenate([in_ids, out_ids], axis=0)
            if union_ids.size > 0:
                m_x = sig_np[union_ids, j, :A_use].mean(axis=0)
            else:
                m_x = np.full((A_use,), 1e-12, dtype=np.float64)
            mean_ref_x[j, :] = np.maximum(m_x, 1e-12)
            if out_ids.size > 0:
                m_z = sig_np[out_ids, j, :A_use].mean(axis=0)
            else:
                m_z = np.full((A_use,), 1e-12, dtype=np.float64)
            mean_ref_z[j, :] = np.maximum(m_z, 1e-12)

    # ---- Step 2: GPU-vectorized per-x scoring ----
    if P_full == 0:
        return np.zeros(N, dtype=np.float32), labels

    dev = torch.device(device)
    eps = 1e-12

    # Slice to target row of sig + first A_use augmentations. (N, A_use) fp32.
    sig_target = torch.from_numpy(
        sig_np[int(target_model), :, :A_use].astype(np.float32, copy=False)
    ).to(dev)
    mean_ref_x_t = torch.from_numpy(mean_ref_x.astype(np.float32, copy=False)).to(dev)  # (N, A_use)
    mean_ref_z_t = torch.from_numpy(mean_ref_z.astype(np.float32, copy=False)).to(dev)  # (N, A_use)
    pop_idx = torch.from_numpy(pop_all.astype(np.int64, copy=False)).to(dev)            # (P,)

    # Precompute prob_ratio_z_rev[p, k] = (mean_z_OUT_only + eps) / (s_z + eps)
    s_z = sig_target[pop_idx, :] + eps                              # (P, A_use)
    mean_z = mean_ref_z_t[pop_idx, :] + eps                         # (P, A_use)  ← OUT-only
    prob_ratio_z_rev = mean_z / s_z                                  # (P, A_use)

    # Per-x ratio: prob_ratio_x[n, k] = (s_x + eps) / (mean_x_INplusOUT + eps)
    prob_ratio_x_all = (sig_target + eps) / (mean_ref_x_t + eps)     # (N, A_use)

    raw_all = torch.zeros(N, dtype=torch.float32, device=dev)
    half_threshold = float(A_use) / 2.0

    for chunk_start in range(0, N, int(chunk_size)):
        chunk_end = min(chunk_start + int(chunk_size), N)
        x_chunk = prob_ratio_x_all[chunk_start:chunk_end, :]         # (C, A_use)

        # score[c, p, k] = x_chunk[c, k] * prob_ratio_z_rev[p, k]
        score = x_chunk[:, None, :] * prob_ratio_z_rev[None, :, :]   # (C, P, A_use)
        votes_bool = score > float(gamma)                            # (C, P, A_use)

        if use_augmented:
            # majority[c, p] = (sum over A of votes_bool[c, p, :]) > A/2
            vote_count = votes_bool.sum(dim=2)                       # (C, P) int
            majority = vote_count.float() > half_threshold           # (C, P) bool
        else:
            majority = votes_bool[:, :, 0]                           # (C, P) bool

        raw_all[chunk_start:chunk_end] = majority.float().mean(dim=1)

    member_score = raw_all.cpu().numpy().astype(np.float32)

    # Overlap correction for non-member x (matches the numpy version's per-x branch).
    if P_full > 1:
        non_member_mask = ~labels
        if non_member_mask.any():
            ms = member_score[non_member_mask].astype(np.float64)
            ms = (ms * P_full - 1.0) / (P_full - 1.0)
            member_score[non_member_mask] = ms.astype(np.float32)

    return member_score, labels


# -----------------------------
# LOSSAttack
# -----------------------------
def loss_scores_for_model(
    target_model: int,
    logits_for_target: np.ndarray,
    labels: np.ndarray,
    keep_matrix: np.ndarray,
    nb_augmentation: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LOSS attack baseline (target-only): per-sample CE averaged across augmentations,
    negated so higher => more member-like.
    """
    y = np.asarray(labels, dtype=np.int64)
    log_np = np.asarray(logits_for_target, dtype=np.float64)

    # Normalize shapes to (N, A, C).
    if log_np.ndim == 4:
        if log_np.shape[1] == 1:
            log_np = log_np[:, 0, :, :]
        else:
            N0 = log_np.shape[0]
            log_np = log_np.reshape(N0, -1, log_np.shape[-1])

    if log_np.ndim == 2:
        log_np = log_np[:, None, :]

    if log_np.ndim != 3:
        raise ValueError(f"Unexpected logits shape for LOSSAttack: {log_np.shape}")

    N, A, C = log_np.shape
    A_use = min(int(nb_augmentation), A) if nb_augmentation is not None else A

    losses = []
    for a in range(A_use):
        losses.append(ce_loss_from_logits(log_np[:, a, :], y))

    avg_loss = np.mean(np.stack(losses, axis=0), axis=0)
    member_score = (-avg_loss).astype(np.float32)

    mem_labels = keep_matrix[target_model].astype(bool)
    return member_score, mem_labels


# -----------------------------
# evaluation helpers
# -----------------------------
def compute_metrics(
    fpr_list: np.ndarray,
    tpr_list: np.ndarray,
    thresholds: np.ndarray,
    target_fprs: List[float],
):
    tpr_dict = {}
    threshold_dict = {}

    roc_auc = auc(fpr_list, tpr_list)
    acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)

    for target_fpr in target_fprs:
        idx = np.where(fpr_list <= target_fpr)[0]
        if len(idx) > 0:
            tpr_dict[target_fpr] = tpr_list[idx[-1]]
            threshold_dict[target_fpr] = thresholds[idx[-1]]
        else:
            tpr_dict[target_fpr] = 0.0
            threshold_dict[target_fpr] = float("inf")

    return roc_auc, acc, tpr_dict, threshold_dict


def save_predicted_members_from_member_score(member_score: np.ndarray, threshold: float, output_path: str):
    predicted_members = np.where(member_score >= threshold)[0]
    np.save(output_path, predicted_members)


def save_attack_scores(scores: np.ndarray, labels: np.ndarray, output_prefix: str):
    np.save(output_prefix + "_scores.npy", scores.astype(np.float32))
    np.save(output_prefix + "_labels.npy", labels.astype(np.int8))


def save_thresholds(threshold_dict: dict, output_path: str):
    np.save(output_path, threshold_dict, allow_pickle=True)


def save_sample_indices(n_data_points: int, output_path: str):
    np.save(output_path, np.arange(n_data_points, dtype=np.int32))


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run membership inference attacks")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    overrides = parse_overrides(args.override)
    recursive_update(config, overrides)

    required_keys = ["experiment.checkpoint_dir"]
    try:
        validate_config(config, required_keys)
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)

    experiment_dir = config.get("experiment", {}).get("checkpoint_dir", "none")
    train_config_path = os.path.join(experiment_dir, "train_config.yaml")

    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)

    recursive_update(config, train_config)

    aggregated_dir = os.path.join(experiment_dir, "aggregated_results")
    os.makedirs(aggregated_dir, exist_ok=True)

    log_level_str = config.get("experiment", {}).get("log_level", "info").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger("Attack", os.path.join(experiment_dir, "attack_log.log"), level=log_level)

    seed = int(config.get("seed", 42))
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    save_config(config, os.path.join(experiment_dir, "attack_config.yaml"))
    logger.info(f"Saved configuration to {experiment_dir}/attack_config.yaml")

    dataset, _, _, _ = load_dataset(config, mode="inference")
    all_labels = np.array(dataset.targets)

    # Apply the experiment's saved deterministic subsample (if any) so attack
    # signals align with keep_indices over the (possibly subsampled) train pool.
    ds_cfg_attack = config.get("dataset", {})
    n_data_points_cfg = ds_cfg_attack.get("n_data_points", None)
    subsample_seed_attack = int(ds_cfg_attack.get("subsample_seed", config.get("seed", 42)))
    n_full_pool = len(all_labels)
    sub_idx = resolve_subsample(
        n_full_pool, n_data_points_cfg, subsample_seed_attack,
        save_path=os.path.join(experiment_dir, "subsample_indices.npy"),
    )
    if sub_idx.size != n_full_pool:
        all_labels = all_labels[sub_idx]
        logger.info(f"[subsample] {sub_idx.size}/{n_full_pool} (seed={subsample_seed_attack})")

    # Top-level `n_data_points` (legacy) acts as a cap *after* subsampling.
    legacy_cap = int(config.get("n_data_points", len(all_labels)))
    legacy_cap = min(legacy_cap, len(all_labels))
    if legacy_cap < len(all_labels):
        all_labels = all_labels[:legacy_cap]
    n_data_points = int(len(all_labels))
    logger.info(f"Dataset loaded with {n_data_points} points")

    labels_t = torch.from_numpy(all_labels).long()

    # Prefer top-level num_shadow_models from attack config; fall back to train config.
    num_shadow_models = int(config.get("num_shadow_models", config.get("training", {}).get("num_shadow_models", 256)))
    logger.info(f"Using {num_shadow_models} total models")

    keep_indices_path = os.path.join(experiment_dir, "keep_indices.npy")
    keep_indices = np.load(keep_indices_path)
    keep_indices = keep_indices[:num_shadow_models, :n_data_points]
    keep_matrix = keep_indices.astype(bool)

    target_models = [int(x) for x in config.get("target_models", [0])]
    target_fprs = config.get("target_fprs", [0.001, 0.00001])

    ref_pool = _build_reference_pool(num_shadow_models, target_models)
    logger.info(f"Targets ({len(target_models)}): {sorted(target_models)}")
    logger.info(f"Reference pool size: {len(ref_pool)}")

    lira_cfg = config.get("lira", {})
    attackr_cfg = config.get("attackr", {})
    rmia_cfg = config.get("rmia", {})
    loss_cfg = config.get("loss", {})

    lira_nb_aug = int(lira_cfg.get("nb_augmentation", 18))
    lira_fix_variance = bool(lira_cfg.get("fix_variance", False))
    lira_overwrite_signals = bool(lira_cfg.get("overwrite_signals", False))
    lira_num_refs_list = lira_cfg.get("num_ref_models", [256, 64, 16])
    if not isinstance(lira_num_refs_list, list):
        lira_num_refs_list = [lira_num_refs_list]
    lira_ref_seed = int(lira_cfg.get("ref_selection_seed", seed))
    # Vectorized (numpy) LiRA scoring; ~20-30x faster than the per-sample
    # Python-loop reference. Numerically equivalent. Disable to force the
    # reference path (e.g. for cross-validation against an external impl).
    lira_use_fast = bool(lira_cfg.get("use_fast", False))

    attackr_nb_aug = int(attackr_cfg.get("nb_augmentation", 1))
    attackr_num_refs_list = attackr_cfg.get("num_ref_models", [128, 32, 8, 4, 3, 2])
    if not isinstance(attackr_num_refs_list, list):
        attackr_num_refs_list = [attackr_num_refs_list]
    attackr_ref_seed = int(attackr_cfg.get("ref_selection_seed", seed))
    attackr_overwrite_signals = bool(attackr_cfg.get("overwrite_signals", False))
    # Faster Attack-R: pre-slices reference tensors and inlines the empirical
    # CDF interp (skips per-sample np.concatenate / np.linspace / np.interp
    # overhead). Numerically equivalent to the reference; per-sample seeded
    # RNG preserved. Mean speedup ~3-5x at our scale.
    attackr_use_fast = bool(attackr_cfg.get("use_fast", False))

    rmia_nb_aug = int(rmia_cfg.get("nb_augmentation", 18))
    rmia_gamma = float(rmia_cfg.get("gamma", 2.0))
    rmia_pct = float(rmia_cfg.get("proportiontocut", 0.2))

    rmia_top_k = int(rmia_cfg.get("top_k", -1))
    rmia_num_refs_list = rmia_cfg.get("num_ref_models", [256, 64, 16, 4, 2])
    if not isinstance(rmia_num_refs_list, list):
        rmia_num_refs_list = [rmia_num_refs_list]
    rmia_ref_seed = int(rmia_cfg.get("ref_selection_seed", seed))

    rmia_offline = rmia_cfg.get("offline", False)
    rmia_offline_a = float(rmia_cfg.get("offline_a", 0.3))

    rmia_aug_mode = str(rmia_cfg.get("augmentation", "augmented")).lower()
    rmia_signal_name = str(rmia_cfg.get("signal", "sm_taylor_softmax_relative"))

    rmia_extra = {
        "taylor_m": float(rmia_cfg.get("taylor_m", 0.6)),
        "taylor_n": int(rmia_cfg.get("taylor_n", 4)),
    }
    rmia_temp = float(rmia_cfg.get("temperature", 2.0))
    rmia_overwrite_signals = bool(rmia_cfg.get("overwrite_signals", False))
    # GPU fast path for the per-x scoring inner loop (~30-50x faster than the
    # numpy implementation; numerically equivalent for top_k=-1). Set to true
    # only after validating the GPU result against a known-good numpy run on
    # the same cached signals.
    rmia_use_gpu = bool(rmia_cfg.get("use_gpu", False))
    rmia_gpu_chunk_size = int(rmia_cfg.get("gpu_chunk_size", 5000))

    loss_nb_aug = int(loss_cfg.get("nb_augmentation", 1))
    loss_enabled = bool(loss_cfg.get("enabled", True))

    # Per-attack on/off switches.  When false, the attack's signal tensor is
    # never loaded/computed and its per-target loop is skipped entirely —
    # used to re-score a single attack (e.g. Attack-R only) without paying
    # the LiRA/RMIA signal-loading cost.
    lira_enabled    = bool(lira_cfg.get("enabled", True))
    rmia_enabled    = bool(rmia_cfg.get("enabled", True))
    attackr_enabled = bool(attackr_cfg.get("enabled", True))

    signals_lira = None
    signals_rmia = None
    signals_attackr = None

    signal_dir = os.path.join(experiment_dir, "signals")
    os.makedirs(signal_dir, exist_ok=True)

    max_aug_lira = max(lira_nb_aug, 1)
    max_aug_rmia = rmia_nb_aug
    max_aug_attackr = max(attackr_nb_aug, 1)

    logger.info("Streaming over models to compute/load signals...")
    for i in range(num_shadow_models):
        logits_path = os.path.join(experiment_dir, f"model_{i}", "logits", "logits.npy")

        lira_signal_path = os.path.join(signal_dir, f"lira_model_{i}.npy")
        rmia_signal_path = os.path.join(signal_dir, f"rmia_model_{i}.npy")
        attackr_signal_path = os.path.join(signal_dir, f"attackr_model_{i}.npy")

        # If an attack is disabled, treat its signal as "not needed" so the
        # loader never opens its tensor and the per-target loop later short-
        # circuits.  This lets a single-attack rerun avoid touching the
        # other attacks' signal files entirely.
        lira_exists    = lira_enabled    and os.path.exists(lira_signal_path)
        rmia_exists    = rmia_enabled    and os.path.exists(rmia_signal_path)
        attackr_exists = attackr_enabled and os.path.exists(attackr_signal_path)

        need_lira_compute    = lira_enabled    and (lira_overwrite_signals    or not lira_exists)
        need_rmia_compute    = rmia_enabled    and (rmia_overwrite_signals    or not rmia_exists)
        need_attackr_compute = attackr_enabled and (attackr_overwrite_signals or not attackr_exists)

        if (not need_lira_compute) and (not need_rmia_compute) and (not need_attackr_compute):
            # Cache-hit / disabled path: only touch the signal tensors for
            # attacks that are both enabled and present on disk.
            if lira_enabled and lira_exists:
                lira_i = np.load(lira_signal_path)
                if signals_lira is None:
                    N_i, A_lira = lira_i.shape
                    signals_lira = torch.empty((num_shadow_models, N_i, A_lira), dtype=torch.float32)
                signals_lira[i] = torch.from_numpy(lira_i)
            if rmia_enabled and rmia_exists:
                rmia_i = np.load(rmia_signal_path)
                if signals_rmia is None:
                    N_i, A_rmia = rmia_i.shape
                    signals_rmia = torch.empty((num_shadow_models, N_i, A_rmia), dtype=torch.float32)
                signals_rmia[i] = torch.from_numpy(rmia_i)
            if attackr_enabled and attackr_exists:
                attackr_i = np.load(attackr_signal_path)
                if signals_attackr is None:
                    N_i, A_atk = attackr_i.shape
                    signals_attackr = torch.empty((num_shadow_models, N_i, A_atk), dtype=torch.float32)
                signals_attackr[i] = torch.from_numpy(attackr_i)
            continue

        if not os.path.exists(logits_path):
            logger.warning(f"No logits for model_{i}")

            if lira_exists and (signals_lira is None):
                lira_i = np.load(lira_signal_path)
                N_i, A_lira = lira_i.shape
                signals_lira = torch.empty((num_shadow_models, N_i, A_lira), dtype=torch.float32)
            if lira_exists:
                signals_lira[i] = torch.from_numpy(np.load(lira_signal_path))

            if rmia_exists and (signals_rmia is None):
                rmia_i = np.load(rmia_signal_path)
                N_i, A_rmia = rmia_i.shape
                signals_rmia = torch.empty((num_shadow_models, N_i, A_rmia), dtype=torch.float32)
            if rmia_exists:
                signals_rmia[i] = torch.from_numpy(np.load(rmia_signal_path))

            if attackr_exists and (signals_attackr is None):
                atk_i = np.load(attackr_signal_path)
                N_i, A_atk = atk_i.shape
                signals_attackr = torch.empty((num_shadow_models, N_i, A_atk), dtype=torch.float32)
            if attackr_exists:
                signals_attackr[i] = torch.from_numpy(np.load(attackr_signal_path))

            continue

        try:
            logits_np = np.load(logits_path)
            logits_t = torch.from_numpy(np.array(logits_np, dtype=np.float32))
            logits_t = logits_t[:n_data_points]

            # Normalize logits to (N, A, C).
            if logits_t.ndim == 4:
                if logits_t.shape[1] == 1:
                    logits_t = logits_t[:, 0, :, :]
                else:
                    N0 = logits_t.shape[0]
                    logits_t = logits_t.reshape(N0, -1, logits_t.shape[-1])

            if logits_t.ndim == 2:
                logits_t = logits_t[:, None, :]

            if logits_t.ndim != 3:
                raise ValueError(f"Unexpected logits shape for model_{i}: {tuple(logits_t.shape)}")

            N_i, A_i, C_i = logits_t.shape

            if lira_enabled and signals_lira is None:
                A_lira = min(A_i, max_aug_lira)
                if lira_exists and (not need_lira_compute):
                    tmp = np.load(lira_signal_path)
                    _, A_lira = tmp.shape
                signals_lira = torch.empty((num_shadow_models, N_i, A_lira), dtype=torch.float32)

            if rmia_enabled and signals_rmia is None:
                A_rmia = min(A_i, max_aug_rmia)
                if rmia_exists and (not need_rmia_compute):
                    tmp = np.load(rmia_signal_path)
                    _, A_rmia = tmp.shape
                signals_rmia = torch.empty((num_shadow_models, N_i, A_rmia), dtype=torch.float32)

            if attackr_enabled and signals_attackr is None:
                A_atk = min(A_i, max_aug_attackr)
                if attackr_exists and (not need_attackr_compute):
                    tmp = np.load(attackr_signal_path)
                    _, A_atk = tmp.shape
                signals_attackr = torch.empty((num_shadow_models, N_i, A_atk), dtype=torch.float32)

            if lira_exists and (not need_lira_compute):
                signals_lira[i] = torch.from_numpy(np.load(lira_signal_path))
            if rmia_exists and (not need_rmia_compute):
                signals_rmia[i] = torch.from_numpy(np.load(rmia_signal_path))
            if attackr_exists and (not need_attackr_compute):
                signals_attackr[i] = torch.from_numpy(np.load(attackr_signal_path))

            y_np = all_labels[:N_i].astype(np.int64)

            for a in range(A_i):
                logits_2d_t = logits_t[:, a, :]

                if need_lira_compute and a < signals_lira.shape[2]:
                    sig_lira_flat = convert_signals(
                        logits_2d_t, labels_t[:N_i], metric="log-logit-scaling", temp=1.0, extra=None
                    )
                    signals_lira[i, :, a] = sig_lira_flat.cpu()

                if need_rmia_compute and a < signals_rmia.shape[2]:
                    rmia_metric = None
                    rmia_extra_used = None

                    if rmia_signal_name == "softmax_relative":
                        rmia_metric = "softmax"
                        rmia_extra_used = None
                    elif rmia_signal_name == "taylor_softmax_relative":
                        rmia_metric = "taylor"
                        rmia_extra_used = {"taylor_n": rmia_extra["taylor_n"]}
                    elif rmia_signal_name == "sm_softmax_relative":
                        rmia_metric = "soft-margin"
                        rmia_extra_used = {"taylor_m": rmia_extra["taylor_m"]}
                    elif rmia_signal_name == "sm_taylor_softmax_relative":
                        rmia_metric = "taylor-soft-margin"
                        rmia_extra_used = {"taylor_m": rmia_extra["taylor_m"], "taylor_n": rmia_extra["taylor_n"]}
                    else:
                        rmia_metric = rmia_signal_name
                        rmia_extra_used = rmia_extra

                    sig_rmia_flat = convert_signals(
                        logits_2d_t,
                        labels_t[:N_i],
                        metric=rmia_metric,
                        temp=rmia_temp,
                        extra=rmia_extra_used,
                    )
                    signals_rmia[i, :, a] = sig_rmia_flat.cpu()

                if need_attackr_compute and a < signals_attackr.shape[2]:
                    logits_2d_np = logits_2d_t.detach().cpu().numpy()
                    ce = ce_loss_from_logits(logits_2d_np, y_np)
                    signals_attackr[i, :, a] = torch.from_numpy(ce.astype(np.float32))

            if need_lira_compute:
                np.save(lira_signal_path, signals_lira[i].numpy())
            if need_rmia_compute:
                np.save(rmia_signal_path, signals_rmia[i].numpy())
            if need_attackr_compute:
                np.save(attackr_signal_path, signals_attackr[i].numpy())

            del logits_t, logits_np
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing model_{i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Require signals only for the attacks that are still enabled.
    missing = []
    if lira_enabled    and signals_lira    is None: missing.append("LiRA")
    if rmia_enabled    and signals_rmia    is None: missing.append("RMIA")
    if attackr_enabled and signals_attackr is None: missing.append("Attack-R")
    if missing:
        logger.error(f"Missing signals for enabled attacks: {missing}. Aborting.")
        return

    if lira_enabled:    logger.info(f"LiRA signals:    {signals_lira.shape}")
    if rmia_enabled:    logger.info(f"RMIA signals:    {signals_rmia.shape}")
    if attackr_enabled: logger.info(f"Attack-R losses: {signals_attackr.shape}")

    # Align all attack inputs to a common sample count, using only the
    # tensors that exist for the enabled attacks.
    enabled_shapes = [
        int(s.shape[1]) for s in
        (signals_lira if lira_enabled else None,
         signals_rmia if rmia_enabled else None,
         signals_attackr if attackr_enabled else None)
        if s is not None
    ]
    signal_n = min(enabled_shapes) if enabled_shapes else int(keep_matrix.shape[1])
    available_n = min(signal_n, int(keep_matrix.shape[1]), int(all_labels.shape[0]))
    if available_n != n_data_points:
        logger.warning(
            f"Adjusted attack sample count from {n_data_points} to {available_n} "
            f"(labels={len(all_labels)}, keep_matrix={keep_matrix.shape[1]}, signals={signal_n})"
        )

    n_data_points = int(available_n)
    all_labels = all_labels[:n_data_points]
    labels_t = labels_t[:n_data_points]
    keep_matrix = keep_matrix[:, :n_data_points]
    # Trim per-attack signals only if that attack is enabled (and therefore
    # has a populated tensor).  Disabled attacks remain None.
    if signals_lira    is not None: signals_lira    = signals_lira[:, :n_data_points, :]
    if signals_rmia    is not None: signals_rmia    = signals_rmia[:, :n_data_points, :]
    if signals_attackr is not None: signals_attackr = signals_attackr[:, :n_data_points, :]

    attack_runtimes = {"LiRA": {}, "Attack-R": {}, "RMIA": {}, "LOSS": []}

    pooled_scores = {
        "LiRA": {},
        "Attack-R": {},
        "RMIA": {},
        "LOSS": {"scores": [], "labels": []},
    }

    for num_refs in lira_num_refs_list:
        pooled_scores["LiRA"][num_refs] = {"scores": [], "labels": []}
    for num_refs in attackr_num_refs_list:
        pooled_scores["Attack-R"][num_refs] = {"scores": [], "labels": []}
    for num_refs in rmia_num_refs_list:
        pooled_scores["RMIA"][num_refs] = {"scores": [], "labels": []}

    per_model_metrics = {"LiRA": {}, "Attack-R": {}, "RMIA": {}, "LOSS": []}
    for num_refs in lira_num_refs_list:
        per_model_metrics["LiRA"][num_refs] = []
    for num_refs in attackr_num_refs_list:
        per_model_metrics["Attack-R"][num_refs] = []
    for num_refs in rmia_num_refs_list:
        per_model_metrics["RMIA"][num_refs] = []

    pooled_metrics = {}

    runtime_csv = os.path.join(experiment_dir, "runtime.csv")
    runtime_extras = {
        "dataset": str(config.get("dataset", {}).get("name", "")).lower(),
        "arch": config.get("model", {}).get("architecture", ""),
        "loading": str(config.get("dataset", {}).get("loading", "gpu")).lower(),
        "n_data_points": (
            "full" if config.get("dataset", {}).get("n_data_points") is None
            else int(config.get("dataset", {}).get("n_data_points"))
        ),
    }
    target_start_times: dict = {}

    for t in target_models:
        # Outer per-target marker (one INFO line per target). Inner per-attack
        # / per-refs banners and per-target AUC/TPR lines are demoted to DEBUG;
        # the aggregate summary at the end is the canonical INFO output.
        logger.info(f"Target {t}/{target_models[-1]}")

        target_model_dir = os.path.join(experiment_dir, f"model_{t}")
        attack_results_dir = os.path.join(target_model_dir, "attack_results")
        os.makedirs(attack_results_dir, exist_ok=True)

        save_sample_indices(n_data_points, os.path.join(attack_results_dir, "sample_indices.npy"))
        target_start_times[int(t)] = time.perf_counter()

        # ========== LiRA ==========
        logger.debug("Running LiRA with different num_ref_models"
                     if lira_enabled else "LiRA disabled; skipping")

        signals_lira_subset = (
            signals_lira[:, :, :lira_nb_aug] if lira_enabled else None
        )

        for num_refs in (lira_num_refs_list if lira_enabled else []):
            lira_prefix = os.path.join(attack_results_dir, f"lira_refs_{num_refs}")
            if (not lira_overwrite_signals) and os.path.exists(lira_prefix + "_scores.npy"):
                logger.debug(f"[LiRA | refs={num_refs}] cached; skipping recompute")
                continue
            start_time = time.time()

            lira_ref_models_t = ref_pool
            if num_refs > 0 and ref_pool.size > num_refs:
                # LT-IQR convention: same M shadows are reused across all targets
                # (one global subsample per (attack, M) configuration).
                lira_ref_models_t = _deterministic_choice(ref_pool, num_refs, seed=int(lira_ref_seed))

            # Per-iteration effective fix_variance: forced True when M < 64
            # (per-sample σ from few refs is too noisy), otherwise uses the
            # config default. Computed per iteration so it is NOT sticky across
            # the num_refs loop — previously, after any M < 64 ran, all later
            # M values inherited fix_variance=True regardless of M.
            lira_fix_variance_iter = lira_fix_variance or (num_refs < 64)

            _lira_fn = lira_scores_for_model_fast if lira_use_fast else lira_scores_for_model
            lira_scores, lira_labels = _lira_fn(
                t, signals_lira_subset, keep_matrix, lira_fix_variance_iter, ref_models=lira_ref_models_t
            )

            elapsed = time.time() - start_time
            if num_refs not in attack_runtimes["LiRA"]:
                attack_runtimes["LiRA"][num_refs] = []
            attack_runtimes["LiRA"][num_refs].append(elapsed)

            fpr_lira, tpr_lira, thresholds_lira = roc_curve(lira_labels.astype(int), lira_scores)
            roc_auc_lira, acc_lira, tpr_dict_lira, threshold_dict_lira = compute_metrics(
                fpr_lira, tpr_lira, thresholds_lira, target_fprs
            )

            logger.debug(f"[LiRA | refs={num_refs}] Model {t} AUC = {roc_auc_lira * 100:.2f}%, Acc = {acc_lira * 100:.2f}%, Runtime = {elapsed:.2f}s")
            tpr_log_parts = [f"TPR@{fpr * 100:.4g}%FPR = {tpr_dict_lira[fpr] * 100:.3f}%" for fpr in target_fprs]
            logger.debug(f"[LiRA | refs={num_refs}] " + ", ".join(tpr_log_parts))

            save_attack_scores(lira_scores, lira_labels.astype(int), lira_prefix)
            save_thresholds(threshold_dict_lira, os.path.join(attack_results_dir, f"lira_refs_{num_refs}_thresholds_by_fpr.npy"))

            pooled_scores["LiRA"][num_refs]["scores"].append(lira_scores)
            pooled_scores["LiRA"][num_refs]["labels"].append(lira_labels.astype(int))
            per_model_metrics["LiRA"][num_refs].append({
                "model": t, "auc": roc_auc_lira, "acc": acc_lira, "tpr_dict": tpr_dict_lira, "runtime": elapsed
            })

        # ========== Attack-R ==========
        logger.debug("Running Attack-R with different num_ref_models"
                     if attackr_enabled else "Attack-R disabled; skipping")

        ce_attackr_subset = (
            signals_attackr[:, :, :attackr_nb_aug] if attackr_enabled else None
        )

        for num_refs in (attackr_num_refs_list if attackr_enabled else []):
            attackr_prefix = os.path.join(attack_results_dir, f"attackr_refs_{num_refs}")
            if (not attackr_overwrite_signals) and os.path.exists(attackr_prefix + "_scores.npy"):
                logger.debug(f"[Attack-R | refs={num_refs}] cached; skipping recompute")
                continue
            start_time = time.time()

            # LT-IQR convention: globally subsample M shadows (same M across
            # all targets and all samples), not per-sample subsampling from
            # the 256-pool.
            attackr_ref_models_t = ref_pool
            if num_refs > 0 and ref_pool.size > num_refs:
                attackr_ref_models_t = _deterministic_choice(
                    ref_pool, num_refs, seed=int(attackr_ref_seed)
                )
            _attackr_fn = attackr_scores_for_model_fast if attackr_use_fast else attackr_scores_for_model
            atk_scores_raw, atk_labels = _attackr_fn(
                t,
                ce_attackr_subset,
                keep_matrix,
                ref_models=attackr_ref_models_t,
                num_ref_models=-1,           # disable per-sample subsampling
                ref_selection_seed=attackr_ref_seed,
            )
            atk_scores = atk_scores_raw.astype(np.float32, copy=False)

            elapsed = time.time() - start_time
            if num_refs not in attack_runtimes["Attack-R"]:
                attack_runtimes["Attack-R"][num_refs] = []
            attack_runtimes["Attack-R"][num_refs].append(elapsed)

            fpr_attackr, tpr_attackr, thresholds_attackr = roc_curve(atk_labels.astype(int), atk_scores)
            roc_auc_attackr, acc_attackr, tpr_dict_attackr, threshold_dict_attackr = compute_metrics(
                fpr_attackr, tpr_attackr, thresholds_attackr, target_fprs
            )

            logger.debug(f"[Attack-R | refs={num_refs}] Model {t} AUC = {roc_auc_attackr * 100:.2f}%, Acc = {acc_attackr * 100:.2f}%, Runtime = {elapsed:.2f}s")
            tpr_log_parts = [f"TPR@{fpr * 100:.4g}%FPR = {tpr_dict_attackr[fpr] * 100:.3f}%" for fpr in target_fprs]
            logger.debug(f"[Attack-R | refs={num_refs}] " + ", ".join(tpr_log_parts))

            save_attack_scores(atk_scores, atk_labels.astype(int), attackr_prefix)
            save_thresholds(threshold_dict_attackr, os.path.join(attack_results_dir, f"attackr_refs_{num_refs}_thresholds_by_fpr.npy"))

            pooled_scores["Attack-R"][num_refs]["scores"].append(atk_scores)
            pooled_scores["Attack-R"][num_refs]["labels"].append(atk_labels.astype(int))
            per_model_metrics["Attack-R"][num_refs].append({
                "model": t, "auc": roc_auc_attackr, "acc": acc_attackr, "tpr_dict": tpr_dict_attackr, "runtime": elapsed
            })

        # ========== RMIA ==========
        logger.debug("Running RMIA with different num_ref_models"
                     if rmia_enabled else "RMIA disabled; skipping")

        signals_rmia_subset = (
            signals_rmia[:, :, :rmia_nb_aug] if rmia_enabled else None
        )

        for num_refs in (rmia_num_refs_list if rmia_enabled else []):
            rmia_prefix = os.path.join(attack_results_dir, f"rmia_refs_{num_refs}")
            if (not rmia_overwrite_signals) and os.path.exists(rmia_prefix + "_scores.npy"):
                logger.debug(f"[RMIA | refs={num_refs}] cached; skipping recompute")
                continue
            start_time = time.time()

            # LT-IQR convention: globally subsample M shadows (same M across
            # all targets and all samples) and pass them as `ref_models`.
            # RMIA's internal `num_ref_models` continues to control the
            # M/2 IN, M/2 OUT slicing per sample (so we set it to 2*M to
            # avoid truncating the available IN/OUT counts within the
            # selected M).
            rmia_ref_models_t = ref_pool
            if num_refs > 0 and ref_pool.size > num_refs:
                rmia_ref_models_t = _deterministic_choice(
                    ref_pool, num_refs, seed=int(rmia_ref_seed)
                )
            _rmia_fn = rmia_scores_for_model_gpu if rmia_use_gpu else rmia_scores_for_model
            _rmia_extra_kw = {"device": "cuda", "chunk_size": rmia_gpu_chunk_size} if rmia_use_gpu else {}
            rmia_scores_raw, rmia_labels = _rmia_fn(
                t,
                signals_rmia_subset,
                keep_matrix,
                ref_models=rmia_ref_models_t,
                gamma=rmia_gamma,
                proportiontocut=rmia_pct,
                top_k=rmia_top_k,
                num_ref_models=2 * int(num_refs),  # never truncate IN/OUT slices
                ref_selection_seed=rmia_ref_seed,
                offline=rmia_offline,
                offline_a=rmia_offline_a,
                augmentation_mode=rmia_aug_mode,
                **_rmia_extra_kw,
            )
            rmia_scores = rmia_scores_raw.astype(np.float32, copy=False)

            elapsed = time.time() - start_time
            if num_refs not in attack_runtimes["RMIA"]:
                attack_runtimes["RMIA"][num_refs] = []
            attack_runtimes["RMIA"][num_refs].append(elapsed)

            fpr_rmia, tpr_rmia, thresholds_rmia = roc_curve(rmia_labels.astype(int), rmia_scores)
            roc_auc_rmia, acc_rmia, tpr_dict_rmia, threshold_dict_rmia = compute_metrics(
                fpr_rmia, tpr_rmia, thresholds_rmia, target_fprs
            )

            logger.debug(f"[RMIA | refs={num_refs}] Model {t} AUC = {roc_auc_rmia * 100:.2f}%, Acc = {acc_rmia * 100:.2f}%, Runtime = {elapsed:.2f}s")
            tpr_log_parts = [f"TPR@{fpr * 100:.4g}%FPR = {tpr_dict_rmia[fpr] * 100:.3f}%" for fpr in target_fprs]
            logger.debug(f"[RMIA | refs={num_refs}] " + ", ".join(tpr_log_parts))

            save_attack_scores(rmia_scores, rmia_labels.astype(int), rmia_prefix)
            save_thresholds(threshold_dict_rmia, os.path.join(attack_results_dir, f"rmia_refs_{num_refs}_thresholds_by_fpr.npy"))

            pooled_scores["RMIA"][num_refs]["scores"].append(rmia_scores)
            pooled_scores["RMIA"][num_refs]["labels"].append(rmia_labels.astype(int))
            per_model_metrics["RMIA"][num_refs].append({
                "model": t, "auc": roc_auc_rmia, "acc": acc_rmia, "tpr_dict": tpr_dict_rmia, "runtime": elapsed
            })

        # ========== LOSS ==========
        if loss_enabled:
            logger.debug("Running LOSS Attack")

            logits_path_t = os.path.join(experiment_dir, f"model_{t}", "logits", "logits.npy")
            if not os.path.exists(logits_path_t):
                logger.warning(f"[LOSS] Missing logits for model_{t}. Skipping.")
            else:
                try:
                    start_time = time.time()

                    logits_np_t = np.load(logits_path_t)
                    logits_np_t = np.asarray(logits_np_t)[:n_data_points]

                    loss_scores, loss_labels = loss_scores_for_model(
                        t, logits_np_t, all_labels, keep_matrix=keep_matrix, nb_augmentation=loss_nb_aug
                    )

                    elapsed = time.time() - start_time
                    attack_runtimes["LOSS"].append(elapsed)

                    fpr_loss, tpr_loss, thresholds_loss = roc_curve(loss_labels.astype(int), loss_scores)
                    roc_auc_loss, acc_loss, tpr_dict_loss, threshold_dict_loss = compute_metrics(
                        fpr_loss, tpr_loss, thresholds_loss, target_fprs
                    )

                    logger.debug(f"[LOSS] Model {t} AUC = {roc_auc_loss * 100:.2f}%, Acc = {acc_loss * 100:.2f}%, Runtime = {elapsed:.2f}s")
                    tpr_log_parts = [f"TPR@{fpr * 100:.4g}%FPR = {tpr_dict_loss[fpr] * 100:.3f}%" for fpr in target_fprs]
                    logger.debug(f"[LOSS] " + ", ".join(tpr_log_parts))

                    loss_prefix = os.path.join(attack_results_dir, "loss")
                    save_attack_scores(loss_scores, loss_labels.astype(int), loss_prefix)
                    save_thresholds(threshold_dict_loss, os.path.join(attack_results_dir, "loss_thresholds_by_fpr.npy"))

                    pooled_scores["LOSS"]["scores"].append(loss_scores)
                    pooled_scores["LOSS"]["labels"].append(loss_labels.astype(int))
                    per_model_metrics["LOSS"].append({
                        "model": t, "auc": roc_auc_loss, "acc": acc_loss, "tpr_dict": tpr_dict_loss, "runtime": elapsed
                    })

                except Exception as e:
                    logger.error(f"[LOSS] Error: {e}")
                    import traceback
                    traceback.print_exc()

        # End of per-target attack work — append to unified runtime.csv.
        append_runtime(
            runtime_csv, phase="attacks", item=f"target_{t}",
            seconds=time.perf_counter() - target_start_times[int(t)],
            **runtime_extras,
        )

    # ========== SUMMARY ==========
    logger.info("\n" + "=" * 60)
    logger.info("ATTACK RUNTIME STATISTICS")
    logger.info("=" * 60)

    runtime_summary_csv = os.path.join(aggregated_dir, "runtime_summary.csv")
    with open(runtime_summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack", "num_refs", "avg_runtime_sec", "std_runtime_sec"])

        for num_refs in lira_num_refs_list:
            runtimes = attack_runtimes["LiRA"].get(num_refs, [])
            if not runtimes:
                logger.info(f"[LiRA | refs={num_refs}] all targets cached; no runtime recorded")
                writer.writerow(["LiRA", num_refs, "cached", "cached"])
                continue
            avg_rt = np.mean(runtimes)
            std_rt = np.std(runtimes)
            logger.info(f"[LiRA | refs={num_refs}] Avg Runtime: {avg_rt:.2f}s ± {std_rt:.2f}s")
            writer.writerow(["LiRA", num_refs, avg_rt, std_rt])

        for num_refs in attackr_num_refs_list:
            runtimes = attack_runtimes["Attack-R"].get(num_refs, [])
            if not runtimes:
                logger.info(f"[Attack-R | refs={num_refs}] all targets cached; no runtime recorded")
                writer.writerow(["Attack-R", num_refs, "cached", "cached"])
                continue
            avg_rt = np.mean(runtimes)
            std_rt = np.std(runtimes)
            logger.info(f"[Attack-R | refs={num_refs}] Avg Runtime: {avg_rt:.2f}s ± {std_rt:.2f}s")
            writer.writerow(["Attack-R", num_refs, avg_rt, std_rt])

        for num_refs in rmia_num_refs_list:
            runtimes = attack_runtimes["RMIA"].get(num_refs, [])
            if not runtimes:
                logger.info(f"[RMIA | refs={num_refs}] all targets cached; no runtime recorded")
                writer.writerow(["RMIA", num_refs, "cached", "cached"])
                continue
            avg_rt = np.mean(runtimes)
            std_rt = np.std(runtimes)
            logger.info(f"[RMIA | refs={num_refs}] Avg Runtime: {avg_rt:.2f}s ± {std_rt:.2f}s")
            writer.writerow(["RMIA", num_refs, avg_rt, std_rt])

        if attack_runtimes["LOSS"]:
            runtimes = attack_runtimes["LOSS"]
            avg_rt = np.mean(runtimes)
            std_rt = np.std(runtimes)
            logger.info(f"[LOSS] Avg Runtime: {avg_rt:.2f}s ± {std_rt:.2f}s")
            writer.writerow(["LOSS", "N/A", avg_rt, std_rt])

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY ACROSS ALL TARGET MODELS")
    logger.info("=" * 60)

    for num_refs in lira_num_refs_list:
        entries = per_model_metrics["LiRA"][num_refs]
        if not entries:
            continue

        aucs = np.array([e["auc"] for e in entries])
        accs = np.array([e["acc"] for e in entries])
        runtimes = np.array([e["runtime"] for e in entries])
        tpr_by_fpr = {fpr: [] for fpr in target_fprs}
        for e in entries:
            for fpr in target_fprs:
                tpr_by_fpr[fpr].append(e["tpr_dict"].get(fpr, 0.0))

        logger.info(f"\n[LiRA | refs={num_refs}] Average over {len(aucs)} models:")
        logger.info(f"  AUC: {np.mean(aucs) * 100:.2f}% ± {np.std(aucs) * 100:.2f}%")
        logger.info(f"  Acc: {np.mean(accs) * 100:.2f}% ± {np.std(accs) * 100:.2f}%")
        logger.info(f"  Runtime: {np.mean(runtimes):.2f}s ± {np.std(runtimes):.2f}s")
        for fpr in target_fprs:
            tpr_vals = np.array(tpr_by_fpr[fpr], dtype=float)
            logger.info(f"  TPR@{fpr * 100:.4g}%FPR: {np.mean(tpr_vals) * 100:.3f}% ± {np.std(tpr_vals) * 100:.3f}%")

        per_model_csv = os.path.join(aggregated_dir, f"lira_refs_{num_refs}_per_model.csv")
        with open(per_model_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["model_id", "auc", "acc", "runtime"] + [f"tpr_at_fpr_{fpr}" for fpr in target_fprs]
            writer.writerow(header)
            for e in entries:
                row = [e["model"], e["auc"], e["acc"], e["runtime"]] + [e["tpr_dict"].get(fpr, 0.0) for fpr in target_fprs]
                writer.writerow(row)

        agg_csv = os.path.join(aggregated_dir, f"lira_refs_{num_refs}_aggregate.csv")
        with open(agg_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["mean_auc", float(np.mean(aucs))])
            writer.writerow(["std_auc", float(np.std(aucs))])
            writer.writerow(["mean_acc", float(np.mean(accs))])
            writer.writerow(["std_acc", float(np.std(accs))])
            writer.writerow(["mean_runtime", float(np.mean(runtimes))])
            writer.writerow(["std_runtime", float(np.std(runtimes))])
            for fpr in target_fprs:
                tpr_vals = np.array(tpr_by_fpr[fpr], dtype=float)
                writer.writerow([f"mean_tpr_at_fpr_{fpr}", float(np.mean(tpr_vals))])
                writer.writerow([f"std_tpr_at_fpr_{fpr}", float(np.std(tpr_vals))])

    for num_refs in attackr_num_refs_list:
        entries = per_model_metrics["Attack-R"][num_refs]
        if not entries:
            continue

        aucs = np.array([e["auc"] for e in entries])
        accs = np.array([e["acc"] for e in entries])
        runtimes = np.array([e["runtime"] for e in entries])
        tpr_by_fpr = {fpr: [] for fpr in target_fprs}
        for e in entries:
            for fpr in target_fprs:
                tpr_by_fpr[fpr].append(e["tpr_dict"].get(fpr, 0.0))

        logger.info(f"\n[Attack-R | refs={num_refs}] Average over {len(aucs)} models:")
        logger.info(f"  AUC: {np.mean(aucs) * 100:.2f}% ± {np.std(aucs) * 100:.2f}%")
        logger.info(f"  Acc: {np.mean(accs) * 100:.2f}% ± {np.std(accs) * 100:.2f}%")
        logger.info(f"  Runtime: {np.mean(runtimes):.2f}s ± {np.std(runtimes):.2f}s")
        for fpr in target_fprs:
            tpr_vals = np.array(tpr_by_fpr[fpr], dtype=float)
            logger.info(f"  TPR@{fpr * 100:.4g}%FPR: {np.mean(tpr_vals) * 100:.3f}% ± {np.std(tpr_vals) * 100:.3f}%")

        per_model_csv = os.path.join(aggregated_dir, f"attackr_refs_{num_refs}_per_model.csv")
        with open(per_model_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["model_id", "auc", "acc", "runtime"] + [f"tpr_at_fpr_{fpr}" for fpr in target_fprs]
            writer.writerow(header)
            for e in entries:
                row = [e["model"], e["auc"], e["acc"], e["runtime"]] + [e["tpr_dict"].get(fpr, 0.0) for fpr in target_fprs]
                writer.writerow(row)

        agg_csv = os.path.join(aggregated_dir, f"attackr_refs_{num_refs}_aggregate.csv")
        with open(agg_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["mean_auc", float(np.mean(aucs))])
            writer.writerow(["std_auc", float(np.std(aucs))])
            writer.writerow(["mean_acc", float(np.mean(accs))])
            writer.writerow(["std_acc", float(np.std(accs))])
            writer.writerow(["mean_runtime", float(np.mean(runtimes))])
            writer.writerow(["std_runtime", float(np.std(runtimes))])
            for fpr in target_fprs:
                tpr_vals = np.array(tpr_by_fpr[fpr], dtype=float)
                writer.writerow([f"mean_tpr_at_fpr_{fpr}", float(np.mean(tpr_vals))])
                writer.writerow([f"std_tpr_at_fpr_{fpr}", float(np.std(tpr_vals))])

    for num_refs in rmia_num_refs_list:
        entries = per_model_metrics["RMIA"][num_refs]
        if not entries:
            continue

        aucs = np.array([e["auc"] for e in entries])
        accs = np.array([e["acc"] for e in entries])
        runtimes = np.array([e["runtime"] for e in entries])
        tpr_by_fpr = {fpr: [] for fpr in target_fprs}
        for e in entries:
            for fpr in target_fprs:
                tpr_by_fpr[fpr].append(e["tpr_dict"].get(fpr, 0.0))

        logger.info(f"\n[RMIA | refs={num_refs}] Average over {len(aucs)} models:")
        logger.info(f"  AUC: {np.mean(aucs) * 100:.2f}% ± {np.std(aucs) * 100:.2f}%")
        logger.info(f"  Acc: {np.mean(accs) * 100:.2f}% ± {np.std(accs) * 100:.2f}%")
        logger.info(f"  Runtime: {np.mean(runtimes):.2f}s ± {np.std(runtimes):.2f}s")
        for fpr in target_fprs:
            tpr_vals = np.array(tpr_by_fpr[fpr], dtype=float)
            logger.info(f"  TPR@{fpr * 100:.4g}%FPR: {np.mean(tpr_vals) * 100:.3f}% ± {np.std(tpr_vals) * 100:.3f}%")

        per_model_csv = os.path.join(aggregated_dir, f"rmia_refs_{num_refs}_per_model.csv")
        with open(per_model_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["model_id", "auc", "acc", "runtime"] + [f"tpr_at_fpr_{fpr}" for fpr in target_fprs]
            writer.writerow(header)
            for e in entries:
                row = [e["model"], e["auc"], e["acc"], e["runtime"]] + [e["tpr_dict"].get(fpr, 0.0) for fpr in target_fprs]
                writer.writerow(row)

        agg_csv = os.path.join(aggregated_dir, f"rmia_refs_{num_refs}_aggregate.csv")
        with open(agg_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["mean_auc", float(np.mean(aucs))])
            writer.writerow(["std_auc", float(np.std(aucs))])
            writer.writerow(["mean_acc", float(np.mean(accs))])
            writer.writerow(["std_acc", float(np.std(accs))])
            writer.writerow(["mean_runtime", float(np.mean(runtimes))])
            writer.writerow(["std_runtime", float(np.std(runtimes))])
            for fpr in target_fprs:
                tpr_vals = np.array(tpr_by_fpr[fpr], dtype=float)
                writer.writerow([f"mean_tpr_at_fpr_{fpr}", float(np.mean(tpr_vals))])
                writer.writerow([f"std_tpr_at_fpr_{fpr}", float(np.std(tpr_vals))])

    if per_model_metrics["LOSS"]:
        entries = per_model_metrics["LOSS"]
        aucs = np.array([e["auc"] for e in entries])
        accs = np.array([e["acc"] for e in entries])
        runtimes = np.array([e["runtime"] for e in entries])
        tpr_by_fpr = {fpr: [] for fpr in target_fprs}
        for e in entries:
            for fpr in target_fprs:
                tpr_by_fpr[fpr].append(e["tpr_dict"].get(fpr, 0.0))

        logger.info(f"\n[LOSS] Average over {len(aucs)} models:")
        logger.info(f"  AUC: {np.mean(aucs) * 100:.2f}% ± {np.std(aucs) * 100:.2f}%")
        logger.info(f"  Acc: {np.mean(accs) * 100:.2f}% ± {np.std(accs) * 100:.2f}%")
        logger.info(f"  Runtime: {np.mean(runtimes):.2f}s ± {np.std(runtimes):.2f}s")
        for fpr in target_fprs:
            tpr_vals = np.array(tpr_by_fpr[fpr], dtype=float)
            logger.info(f"  TPR@{fpr * 100:.4g}%FPR: {np.mean(tpr_vals) * 100:.3f}% ± {np.std(tpr_vals) * 100:.3f}%")

        per_model_csv = os.path.join(aggregated_dir, "loss_per_model.csv")
        with open(per_model_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["model_id", "auc", "acc", "runtime"] + [f"tpr_at_fpr_{fpr}" for fpr in target_fprs]
            writer.writerow(header)
            for e in entries:
                row = [e["model"], e["auc"], e["acc"], e["runtime"]] + [e["tpr_dict"].get(fpr, 0.0) for fpr in target_fprs]
                writer.writerow(row)

        agg_csv = os.path.join(aggregated_dir, "loss_aggregate.csv")
        with open(agg_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["mean_auc", float(np.mean(aucs))])
            writer.writerow(["std_auc", float(np.std(aucs))])
            writer.writerow(["mean_acc", float(np.mean(accs))])
            writer.writerow(["std_acc", float(np.std(accs))])
            writer.writerow(["mean_runtime", float(np.mean(runtimes))])
            writer.writerow(["std_runtime", float(np.std(runtimes))])
            for fpr in target_fprs:
                tpr_vals = np.array(tpr_by_fpr[fpr], dtype=float)
                writer.writerow([f"mean_tpr_at_fpr_{fpr}", float(np.mean(tpr_vals))])
                writer.writerow([f"std_tpr_at_fpr_{fpr}", float(np.std(tpr_vals))])

    logger.info("\n" + "=" * 60)
    logger.info("Attacks completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
