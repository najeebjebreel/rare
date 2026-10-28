"""
Shared scoring and metric utilities for the RARE pipeline.

Contains:
  - Progress-bar helper
  - Precision / Recall @ top-k% against a gold vulnerable set
    (`tail_metrics_at_k`, the single definition used by every table and
    figure generator)
  - LT-IQR, LT-BTM-R and LT-TOP-R trajectory statistics, and the locked
    RARE score
  - Config-override helpers
"""

import sys
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Dataset


# =============================================================================
# Timing utilities
# =============================================================================
def _fmt_time(sec: float) -> str:
    sec = float(max(0.0, sec))
    if sec < 60:
        return f"{sec:.2f}s"
    m = int(sec // 60)
    s = sec - 60 * m
    if m < 60:
        return f"{m:d}m{s:04.1f}s"
    h = int(m // 60)
    m2 = m - 60 * h
    return f"{h:d}h{m2:02d}m{s:04.1f}s"


class ProgressBar:
    def __init__(
        self,
        desc: str,
        total: Optional[int],
        enabled: bool = True,
        width: int = 28,
    ):
        self.desc = str(desc)
        self.total = int(total) if total is not None else None
        self.enabled = bool(enabled)
        self.width = int(width)
        self.t0 = time.time()
        self.last_len = 0
        self.i = 0

    def _render(self, i: int) -> str:
        elapsed = time.time() - self.t0
        if self.total is None or self.total <= 0:
            return f"{self.desc}: {i} | elapsed={_fmt_time(elapsed)}"
        frac = min(1.0, max(0.0, i / float(self.total)))
        done = int(round(frac * self.width))
        bar = "#" * done + "-" * (self.width - done)
        eta = 0.0
        if i > 0 and frac > 0:
            eta = elapsed * (1.0 / frac - 1.0)
        return (
            f"{self.desc}: [{bar}] {i}/{self.total} | elapsed={_fmt_time(elapsed)} | "
            f"eta={_fmt_time(eta)}"
        )

    def update(self, i: int):
        self.i = int(i)
        if not self.enabled:
            return
        s = self._render(self.i)
        pad = " " * max(0, self.last_len - len(s))
        sys.stdout.write("\r" + s + pad)
        sys.stdout.flush()
        self.last_len = len(s)

    def close(self, final_msg: Optional[str] = None):
        if not self.enabled:
            return
        if final_msg is None:
            final_msg = self._render(self.i)
        pad = " " * max(0, self.last_len - len(final_msg))
        sys.stdout.write("\r" + final_msg + pad + "\n")
        sys.stdout.flush()
        self.last_len = 0


class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return x, y, idx


# =============================================================================
# Score-evaluation primitives
# =============================================================================
def _budget_at_frac(k_frac: float, n: int) -> int:
    """Retrieval budget m_k = ceil(k * n) over `n` valid members (Pollock et al.)."""
    return max(0, min(int(np.ceil(float(k_frac) * n)), n))


def tail_metrics_at_k(
    method_scores: np.ndarray,
    keep_mask: np.ndarray,
    k_fracs: List[float],
    *,
    vulnerable: np.ndarray,
    gold_scores: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Top-k% precision and recall of a member ranking against a gold set V.

    With n_mem valid members and m_k = ceil(k * n_mem):
        P@k% = |P_k n V| / m_k        R@k% = |P_k n V| / |V|
    where P_k are the m_k highest-scoring members. The budget is not truncated
    at |V|, so P@k% <= |V| / m_k whenever m_k > |V|.

    Members whose method score (or, when given, gold score) is not finite are
    excluded from n_mem. Returns one row per k with columns
    k, k_samples (= m_k), precision, recall, gold_n (= |V|).
    """
    s_m = np.asarray(method_scores, dtype=np.float32)[keep_mask]
    ok = np.isfinite(s_m)
    if gold_scores is not None:
        ok &= np.isfinite(np.asarray(gold_scores, dtype=np.float32)[keep_mask])
    s_valid = s_m[ok]
    v_valid = np.asarray(vulnerable, dtype=np.int32)[keep_mask][ok]
    n, n_v = s_valid.size, int(v_valid.sum())

    rows = []
    for kf in k_fracs:
        k = _budget_at_frac(kf, n)
        row = {"k": f"{kf * 100:.1f}%", "k_samples": k, "gold_n": n_v}
        if k == 0:
            row.update(precision=0.0, recall=0.0)
        else:
            tp = int(v_valid[np.argpartition(s_valid, -k)[-k:]].sum())
            row.update(precision=tp / k, recall=tp / n_v if n_v > 0 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_lt_iqr_scores(
    losses_NE: np.ndarray,
    keep_mask: np.ndarray,
    start_epoch: int = 0,
    end_epoch: Optional[int] = None,
) -> np.ndarray:
    N, E = losses_NE.shape
    start = max(0, int(start_epoch))
    end = min(E, int(end_epoch) if end_epoch is not None else E)
    if start >= end:
        raise ValueError(f"Invalid epoch range [{start},{end}) for E={E}")
    losses_subset = losses_NE[:, start:end]
    out = np.full(N, np.nan, dtype=np.float32)
    idx = np.where(keep_mask)[0]
    out[idx] = (
        np.quantile(losses_subset[idx], 0.75, axis=1)
        - np.quantile(losses_subset[idx], 0.25, axis=1)
    ).astype(np.float32)
    return out


# =============================================================================
# Locked RARE: z_T(LT-BTM-R) + z_T(LT-IQR)
# =============================================================================
_REF_LOSS_RE = re.compile(r"^ref_losses_E(\d+)_(.+)_ref(\d+)\.npy$")


def _load_target_membership_mask(exp: Path, target_model: int, n: int) -> np.ndarray:
    keep_p = exp / "keep_indices.npy"
    if not keep_p.exists():
        return np.ones(int(n), dtype=bool)
    keep = np.load(keep_p)
    if keep.ndim == 2:
        mask = keep[int(target_model)]
    else:
        mask = keep
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != int(n):
        raise ValueError(
            f"membership mask length {mask.shape[0]} != score length {int(n)} "
            f"for target {target_model}"
        )
    return mask


def _zscore_ignore_nan(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Z-score normalize a score vector while preserving NaNs.

    If `mask` is supplied, mean/std are estimated only on `mask` entries, but
    the resulting calibration is applied to every finite score.
    """
    x = x.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    fit_mask = finite if mask is None else (finite & np.asarray(mask, dtype=bool))
    vals = x[fit_mask]
    if vals.size == 0:
        return np.full(x.shape, np.nan, dtype=np.float32)
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if sd <= 0 or not np.isfinite(sd):
        out[finite] = x[finite] - mu
    else:
        out[finite] = (x[finite] - mu) / sd
    return out.astype(np.float32, copy=False)


def score_ltiqr_trajectory(losses_NE: np.ndarray) -> np.ndarray:
    """LT-IQR over a full target loss trajectory, one score per sample."""
    q = np.quantile(losses_NE, [0.75, 0.25], axis=1)
    return (q[0] - q[1]).astype(np.float32)


def score_avg_min_q_reference_losses(
    reference_losses: List[np.ndarray],
    q: float = 0.50,
) -> np.ndarray:
    """RARE reference-side LT-BTM-R signal.

    For each reference model, average the bottom-q fraction of each sample's
    per-epoch loss trajectory, then average those scores across references.
    """
    if not reference_losses:
        raise ValueError("No reference losses provided for RARE scoring.")
    E = int(reference_losses[0].shape[1])
    k = max(1, int(round(float(q) * E)))
    per_ref = np.stack(
        [
            np.partition(ref_loss, k - 1, axis=1)[:, :k].mean(axis=1)
            for ref_loss in reference_losses
        ],
        axis=0,
    )
    return per_ref.mean(axis=0).astype(np.float32)


def score_avg_max_q_reference_losses(
    reference_losses: List[np.ndarray],
    q: float = 0.50,
) -> np.ndarray:
    """LT-TOP-R: mirror of LT-BTM-R on the top q-fraction of the trajectory.

    For each reference model, average the top-q fraction of each sample's
    per-epoch loss trajectory, then average those scores across references.
    Mirrors `score_avg_min_q_reference_losses` line for line; only the
    partition side differs (`[-k:]` on the upper partition vs `[:k]` on the
    lower), so the q -> k rounding and the cross-reference averaging match.
    """
    if not reference_losses:
        raise ValueError("No reference losses provided for LT-TOP-R scoring.")
    E = int(reference_losses[0].shape[1])
    k = max(1, int(round(float(q) * E)))
    per_ref = np.stack(
        [
            np.partition(ref_loss, -k, axis=1)[:, -k:].mean(axis=1)
            for ref_loss in reference_losses
        ],
        axis=0,
    )
    return per_ref.mean(axis=0).astype(np.float32)


def _find_rare_reference_loss(target_dir: Path, ref_idx: int) -> Tuple[int, str]:
    for p in sorted(target_dir.glob(f"ref_losses_E*_*_ref{int(ref_idx)}.npy")):
        m = _REF_LOSS_RE.match(p.name)
        if m and int(m.group(3)) == int(ref_idx):
            return int(m.group(1)), m.group(2)
    raise FileNotFoundError(f"No RARE reference loss file for ref{ref_idx} in {target_dir}")


def load_rare_locked_scores(
    experiment_dir: str | Path,
    target_model: int,
    num_refs: int = 1,
    q: float = 0.50,
) -> np.ndarray:
    """Load trajectories and compute locked RARE = z_T(LT-BTM-R) + z_T(LT-IQR).

    The target trajectory is read from `per_epoch_losses.npy`; reference
    trajectories are read from `ref_losses_E*_*_ref<r>.npy` files produced by
    `scripts/rare_train_references.py`. The z-score moments are estimated
    over the target model's candidate member set `M` from `keep_indices.npy`.
    """
    exp = Path(experiment_dir)
    target_dir = exp / f"model_{int(target_model)}"
    target_loss_p = target_dir / "per_epoch_losses.npy"
    if not target_loss_p.exists():
        raise FileNotFoundError(target_loss_p)

    target_losses = np.load(target_loss_p).astype(np.float32).T
    reference_losses: List[np.ndarray] = []
    for r in range(int(num_refs)):
        E, tag = _find_rare_reference_loss(target_dir, r)
        ref_loss_p = target_dir / f"ref_losses_E{E}_{tag}_ref{r}.npy"
        ref_losses = np.load(ref_loss_p).astype(np.float32)
        if ref_losses.shape != target_losses.shape:
            raise ValueError(
                f"target loss shape {target_losses.shape} != ref{r} loss shape {ref_losses.shape}"
            )
        reference_losses.append(ref_losses)

    s_R = score_avg_min_q_reference_losses(reference_losses, q=q)
    s_T = score_ltiqr_trajectory(target_losses)
    members = _load_target_membership_mask(exp, int(target_model), int(s_T.shape[0]))
    return (
        _zscore_ignore_nan(s_R, members) + _zscore_ignore_nan(s_T, members)
    ).astype(np.float32)


def parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    """
    Parse CLI overrides like:
      analysis.fpr_threshold=0.001
      analysis.target_models=[0,25,50]
      rare.ref_training.num_references=2
    """
    out: Dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}' (expected key=value)")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

        try:
            v_parsed = yaml.safe_load(v)
        except Exception:
            v_parsed = v

        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v_parsed
    return out


def recursive_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            recursive_update(base[k], v)
        else:
            base[k] = v
    return base


