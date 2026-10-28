"""Score and gold-set loaders shared by the paper table generators.

All paths are relative to an experiment directory `exp` and a target index `t`;
the files are written by the pipeline in `scripts/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rare.analysis.utils import load_rare_locked_scores, tail_metrics_at_k  # noqa: F401  (re-exported)

_ATTACK_PREFIX = {"LiRA": "lira_refs", "Attack-R": "attackr_refs", "RMIA": "rmia_refs"}


def load_lira_gold(exp: Path, t: int, fpr: float):
    """(LiRA-256 scores, member mask, vulnerable mask) at the stored FPR threshold."""
    ar = exp / f"model_{t}" / "attack_results"
    scores = np.load(ar / "lira_refs_256_scores.npy").astype(np.float32)
    labels = np.load(ar / "lira_refs_256_labels.npy").astype(bool)
    thr_dict = np.load(ar / "lira_refs_256_thresholds_by_fpr.npy", allow_pickle=True).item()
    if fpr not in thr_dict:
        raise KeyError(f"FPR={fpr} not in LiRA thresholds (target {t})")
    vulnerable = (scores >= float(thr_dict[fpr])) & labels
    return scores, labels, vulnerable


def load_attack_scores(exp: Path, t: int, attack: str, M: int) -> np.ndarray:
    """LiRA / Attack-R / RMIA scores with M reference models (scripts/attacks.py)."""
    p = exp / f"model_{t}" / "attack_results" / f"{_ATTACK_PREFIX[attack]}_{M}_scores.npy"
    return np.load(p).astype(np.float32)


def load_loss_scores(exp: Path, t: int) -> np.ndarray:
    return np.load(exp / f"model_{t}" / "attack_results" / "loss_scores.npy").astype(np.float32)


def load_liu_mlp(exp: Path, t: int, M: int) -> np.ndarray:
    """LT-MLP scores trained on M additional models (scripts/score_liu_mlp.py)."""
    return np.load(exp / f"model_{t}" / f"liu_mlp_scores_R{M}.npy").astype(np.float32)


def load_lt_iqr(exp: Path, t: int) -> np.ndarray:
    """LT-IQR scores (scripts/score_lt_iqr.py)."""
    return np.load(exp / f"model_{t}" / "lt_iqr_scores.npy").astype(np.float32)


def load_rare(exp: Path, t: int, M: int) -> np.ndarray:
    """RARE = z_T(LT-IQR) + z_T(LT-BTM-R), q=50%, with M reference models."""
    return load_rare_locked_scores(exp, t, num_refs=int(M), q=0.50)
