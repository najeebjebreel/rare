"""Numerically stable softmax / log-sum-exp / cross-entropy on numpy arrays."""

import numpy as np
from scipy.special import softmax, logsumexp


def stable_softmax(logits, axis=-1):
    """Numerically stable softmax (scipy)."""
    return softmax(logits, axis=axis)


def stable_logsumexp(logits, axis=-1, keepdims=False):
    """Numerically stable log-sum-exp (scipy)."""
    return logsumexp(logits, axis=axis, keepdims=keepdims)


def compute_cross_entropy_loss(logits, labels):
    """
    Per-sample cross-entropy from logits.

    Args:
        logits: (N, C) array
        labels: (N,) integer labels

    Returns:
        (N,) array of CE losses.
    """
    N = logits.shape[0]
    lse = stable_logsumexp(logits, axis=1, keepdims=False)
    true_class_logits = logits[np.arange(N), labels]
    return lse - true_class_logits


def _normalize_logits_3d(logits: np.ndarray) -> np.ndarray:
    """
    Normalize logits to shape `[N, A, C]`.

    Accepts:
      - `[N, 1, A, C]` -> squeeze dim 1
      - `[N, A, C]`    -> as-is
    """
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim == 4:
        if x.shape[1] == 1:
            x = np.squeeze(x, axis=1)
        else:
            raise ValueError(f"Unexpected 4D logits shape {x.shape}; expected [N,1,A,C].")
    elif x.ndim != 3:
        raise ValueError(f"Expected logits [N,1,A,C] or [N,A,C], got {x.shape}.")
    return x


def cross_entropy_loss(logits, labels):
    """
    Per-example, per-augmentation cross-entropy.

    Args:
        logits: `[N, 1, A, C]` or `[N, A, C]`
        labels: array-like of length >= N, integer class labels

    Returns:
        `[N, A]` losses.
    """
    x = _normalize_logits_3d(np.asarray(logits))

    N, A, C = x.shape
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape[0] < N:
        raise ValueError(f"Labels length {labels.shape[0]} is smaller than N={N}.")

    max_logits = np.max(x, axis=2, keepdims=True)
    stable = x - max_logits
    exp_logits = np.exp(stable, dtype=np.float64)
    probs = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)

    probs_flat = probs.reshape(N * A, C)
    labels_rep = np.repeat(labels[:N], A)
    true_probs_flat = probs_flat[np.arange(N * A), labels_rep]
    true_probs_flat = np.clip(true_probs_flat, 1e-45, 1.0)

    losses = -np.log(true_probs_flat).reshape(N, A)
    return losses
