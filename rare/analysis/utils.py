"""
Shared utilities for the vulnerability-analysis pipeline.

Contains:
  - Timing/progress-bar helpers
  - Per-method evaluation (Spearman / AUROC / KS / precision-recall@k)
  - Distribution-overlap plotting
  - LT-IQR scoring
  - Predicted-class computation on TRAIN under a target shadow model
  - Config override / validation helpers
"""

import sys
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


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


class TimerRegistry:
    def __init__(self):
        self.rows: List[Dict[str, object]] = []

    def record(self, key: str, wall_sec: float):
        self.rows.append({"component": str(key), "wall_sec": float(wall_sec)})

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary_table(self, prefix: str = "") -> str:
        df = self.dataframe()
        if df.empty:
            return ""
        df2 = df.copy()
        df2["wall_hms"] = df2["wall_sec"].map(_fmt_time)
        lines = [f"\n{'=' * 60}\nRuntime summary {prefix}\n{'=' * 60}"]
        mlen = max(12, int(df2["component"].astype(str).map(len).max()))
        lines.append(f"{'component'.ljust(mlen)}  {'wall'}")
        lines.append(f"{'-' * mlen}  {'-' * 12}")
        for _, r in df2.iterrows():
            lines.append(
                f"{str(r['component']).ljust(mlen)}  {str(r['wall_hms']).rjust(12)}"
            )
        total = float(df2["wall_sec"].sum())
        lines.append(f"{'-' * mlen}  {'-' * 12}")
        lines.append(f"{'TOTAL'.ljust(mlen)}  {_fmt_time(total).rjust(12)}")
        return "\n".join(lines)


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
def spearman_corr(scores: np.ndarray, vulnerable: np.ndarray, keep_mask: np.ndarray) -> float:
    s, v = scores[keep_mask], vulnerable[keep_mask].astype(float)
    ok = np.isfinite(s)
    if not np.any(ok):
        return float("nan")
    return float(stats.spearmanr(s[ok], v[ok])[0])


def auroc_members_only(scores: np.ndarray, vulnerable: np.ndarray, keep_mask: np.ndarray) -> float:
    s, y = scores[keep_mask], vulnerable[keep_mask].astype(np.int32)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(s, method="average")
    sum_ranks_pos = float(ranks[y == 1].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ks_stat_members_only(scores: np.ndarray, vulnerable: np.ndarray, keep_mask: np.ndarray) -> float:
    s, y = scores[keep_mask], vulnerable[keep_mask].astype(bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    a, b = s[y], s[~y]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(stats.ks_2samp(a, b, alternative="two-sided", mode="auto").statistic)


def precision_recall_at_k(
    scores: np.ndarray,
    vulnerable: np.ndarray,
    keep_mask: np.ndarray,
    k_fracs: List[float],
) -> pd.DataFrame:
    """Precision / Coverage (= Recall) at top-k% over members vs binary
    vulnerable label. Kept for legacy callers; new code should use
    `tail_metrics_at_k` which also computes Tail ρ and Jaccard vs LiRA."""
    s, v = scores[keep_mask], vulnerable[keep_mask]
    ok = np.isfinite(s)
    s, v = s[ok], v[ok]
    n, n_v = s.size, int(v.sum())
    order = np.argsort(s)[::-1]
    rows = []
    for kf in k_fracs:
        k = int(round(kf * n))
        k = max(0, min(k, n))
        if k <= 0:
            precision = recall = 0.0
        else:
            top = order[:k]
            tp = int(v[top].sum())
            precision = float(tp / k)
            recall = float(tp / n_v) if n_v > 0 else 0.0
        rows.append(
            {
                "k": f"{kf * 100:.1f}%",
                "k_samples": k,
                "precision": precision,
                "recall": recall,
            }
        )
    return pd.DataFrame(rows)


def tail_metrics_at_k(
    method_scores: np.ndarray,
    keep_mask: np.ndarray,
    k_fracs: List[float],
    *,
    vulnerable: Optional[np.ndarray] = None,
    gold_scores: Optional[np.ndarray] = None,
    pool_member_scores: Optional[List[np.ndarray]] = None,
    gold_topk: bool = False,
) -> pd.DataFrame:
    """Top-k tail metrics for ranking-quality evaluation.

    Three modes selected by the supplied gold reference:

    union mode  (pool_member_scores is not None):
      pool_member_scores: list of score arrays already restricted to members
        (one per pool method after self-exclusion), same ordering as
        method_scores[keep_mask].
      At each k, builds V_M(k) = union of top-k member samples from each
      pool array. All indices are in member-subarray coordinates.
        precision  = |M_k ∩ V_M(k)| / k
        recall     = |M_k ∩ V_M(k)| / |V_M(k)|
        jaccard    = NaN  (|V| varies, generally != k, so size-asymmetric)
        tail_rho   = NaN  (binary gold, no continuous reference ranking)
        gold_n     = |V_M(k)|

    lira_topk mode  (gold_topk=True with gold_scores; pool_member_scores None):
      gold_scores: continuous LiRA scores (over all N samples).
      At each k, V_k = top-k% of LiRA scores among members; |V_k| = k.
        precision  = |M_k ∩ V_k| / k
        recall     = |M_k ∩ V_k| / k       (== precision; |V_k|=k)
        jaccard    = |M_k ∩ V_k| / |M_k ∪ V_k|
        tail_rho   = Spearman(method, gold) on M_k ∩ V_k (intersection only)
        gold_n     = k

    lira mode  (gold_scores and vulnerable are not None):
      vulnerable: fixed binary label (LiRA-thr-defined, over all N samples).
      gold_scores: continuous LiRA scores (over all N samples).
        precision  = |M_k ∩ V| / min(k, |V|)    [k capped at |V|]
        recall     = |M_k ∩ V| / |V|
        jaccard    = NaN  (|V|!=k, size-asymmetric — see lira_topk mode)
        tail_rho   = Spearman(method, gold) on M_k ∩ V (the LiRA-thr
                     vulnerable gold), measuring within-tail rank
                     agreement on the gold members M_k recovers
        gold_n     = |V|

      Note: in lira mode the top-k size is capped at |V|, so the
      `k_samples` column reports min(round(k% · n_members), |V|). This
      prevents Precision from being mechanically ceiling-limited to
      |V|/k_samples when the requested k% exceeds |V|/n_members (e.g.,
      k=10% under FPR=1e-4 where |V| ≈ |members|/20). Compare
      k_samples vs gold_n to see whether the cap fired for a given row.
    """
    s_m = np.asarray(method_scores, dtype=np.float32)[keep_mask]
    ok = np.isfinite(s_m)

    # --- union mode ---
    if pool_member_scores is not None:
        ok_idx = np.where(ok)[0]
        s_valid = s_m[ok]
        n = s_valid.size
        rows: List[Dict[str, Any]] = []
        for kf in k_fracs:
            k = max(0, min(int(round(kf * n)), n))
            row: Dict[str, Any] = {"k": f"{kf * 100:.1f}%", "k_samples": k}
            if k == 0:
                row.update(precision=0.0, recall=0.0, jaccard=float("nan"),
                           tail_rho=float("nan"), gold_n=0)
                rows.append(row)
                continue
            # Method top-k in member-subarray coordinates.
            order_m = np.argpartition(s_valid, -k)[-k:]
            set_m = {int(ok_idx[i]) for i in order_m}
            # Union gold-set: top-k from each pool array.
            gold_set: set = set()
            for ps in pool_member_scores:
                ps_f = np.where(np.isfinite(ps), ps, -np.inf).astype(np.float32)
                k_p = min(k, ps_f.size)
                if k_p > 0:
                    gold_set.update(int(i) for i in np.argpartition(ps_f, -k_p)[-k_p:])
            inter = len(set_m & gold_set)
            row["precision"] = inter / k
            row["recall"] = inter / len(gold_set) if gold_set else 0.0
            # Jaccard is only meaningful when |gold|=k (lira_topk regime).
            # Under union mode |gold| varies and is generally != k, so the
            # Jaccard set-similarity is size-asymmetric — drop it.
            row["jaccard"] = float("nan")
            row["tail_rho"] = float("nan")
            row["gold_n"] = len(gold_set)
            rows.append(row)
        return pd.DataFrame(rows)

    # --- lira_topk mode ---
    if gold_topk:
        if gold_scores is None:
            raise ValueError("gold_topk=True requires gold_scores.")
        g_m = np.asarray(gold_scores, dtype=np.float32)[keep_mask]
        ok2 = ok & np.isfinite(g_m)
        ok_idx = np.where(ok2)[0]
        s_valid = s_m[ok2]
        g_valid = g_m[ok2]
        n = s_valid.size
        rows = []
        for kf in k_fracs:
            k = max(0, min(int(round(kf * n)), n))
            row = {"k": f"{kf * 100:.1f}%", "k_samples": k}
            if k == 0:
                row.update(precision=0.0, recall=0.0, jaccard=float("nan"),
                           tail_rho=float("nan"), gold_n=0)
                rows.append(row)
                continue
            order_m = np.argpartition(s_valid, -k)[-k:]
            order_g = np.argpartition(g_valid, -k)[-k:]
            set_m = {int(i) for i in order_m}
            set_g = {int(i) for i in order_g}
            inter_idx = np.array(sorted(set_m & set_g), dtype=np.intp)
            union_sz = len(set_m | set_g)
            row["precision"] = inter_idx.size / k
            row["recall"] = inter_idx.size / k  # |V_k| == k
            row["jaccard"] = inter_idx.size / union_sz if union_sz > 0 else float("nan")
            row["tail_rho"] = (
                float(stats.spearmanr(s_valid[inter_idx], g_valid[inter_idx])[0])
                if inter_idx.size >= 2 else float("nan")
            )
            row["gold_n"] = k
            rows.append(row)
        return pd.DataFrame(rows)

    # --- lira mode ---
    g_m = np.asarray(gold_scores, dtype=np.float32)[keep_mask] if gold_scores is not None else None
    v_m = np.asarray(vulnerable, dtype=np.int32)[keep_mask] if vulnerable is not None else None
    if g_m is not None:
        ok = ok & np.isfinite(g_m)
    s_valid = s_m[ok]
    g_valid = g_m[ok] if g_m is not None else None
    v_valid = v_m[ok] if v_m is not None else None
    n = s_valid.size
    n_v = int(v_valid.sum()) if v_valid is not None else 0

    rows = []
    for kf in k_fracs:
        k = max(0, min(int(round(kf * n)), n))
        # Cap top-k size at |V| so Precision isn't mechanically limited to
        # |V|/k when the requested k% exceeds the vulnerable-set fraction
        # (the FPR=1e-4, k=10% ceiling case).
        if v_valid is not None and n_v > 0:
            k = min(k, n_v)
        row = {"k": f"{kf * 100:.1f}%", "k_samples": k}
        if k == 0:
            row.update(precision=0.0, recall=0.0, jaccard=float("nan"),
                       tail_rho=float("nan"), gold_n=n_v)
            rows.append(row)
            continue
        order_m = np.argpartition(s_valid, -k)[-k:]
        # Precision / recall vs binary label.
        if v_valid is not None:
            tp = int(v_valid[order_m].sum())
            row["precision"] = tp / k
            row["recall"] = tp / n_v if n_v > 0 else 0.0
        else:
            row["precision"] = float("nan")
            row["recall"] = float("nan")
        row["gold_n"] = n_v
        # Jaccard NOT computed under lira mode: |V|!=k makes the Jaccard
        # set-similarity size-asymmetric. Jaccard is only natural under
        # lira_topk where |V|=k. tail_rho = Spearman(M, LiRA) restricted to
        # M's top-k members that are also in the LiRA-thr gold V.
        row["jaccard"] = float("nan")
        if g_valid is not None and v_valid is not None:
            set_m = {int(i) for i in order_m}
            v_set = {int(i) for i in np.where(v_valid)[0]}
            inter_thr_idx = np.array(sorted(set_m & v_set), dtype=np.intp)
            row["tail_rho"] = (
                float(stats.spearmanr(s_valid[inter_thr_idx], g_valid[inter_thr_idx])[0])
                if inter_thr_idx.size >= 2 else float("nan")
            )
        else:
            row["tail_rho"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def corr_vs_lira(method_scores: np.ndarray, lira_scores: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    aa = method_scores[mask]
    bb = lira_scores[mask]
    v = np.isfinite(aa) & np.isfinite(bb)
    aa, bb = aa[v], bb[v]
    if aa.size == 0:
        return {"spearman": float("nan"), "pearson": float("nan"), "kendall": float("nan")}
    spear = float(stats.spearmanr(aa, bb)[0])
    pear = float(np.corrcoef(aa.astype(np.float64), bb.astype(np.float64))[0, 1]) if aa.size >= 2 else float("nan")
    tau = float(stats.kendalltau(aa, bb, nan_policy="omit")[0])
    return {"spearman": spear, "pearson": pear, "kendall": tau}


def evaluate_method(
    name: str,
    scores: np.ndarray,
    keep_mask: np.ndarray,
    k_fracs: List[float],
    *,
    vulnerable: Optional[np.ndarray] = None,
    gold_scores: Optional[np.ndarray] = None,
    pool_member_scores: Optional[List[np.ndarray]] = None,
    gold_topk: bool = False,
    print_report: bool = True,
) -> Dict[str, object]:
    """Compute tail-focused ranking-quality metrics per top-k.

    Returns dict with keys: name, scores, pr (DataFrame with columns
    k, k_samples, precision, recall, jaccard, tail_rho, gold_n).

    Pass pool_member_scores for union mode, gold_scores + gold_topk=True
    for lira_topk mode, or gold_scores + vulnerable for lira mode. See
    tail_metrics_at_k for full semantics.
    """
    s = scores[keep_mask]
    ok = np.isfinite(s)
    if print_report:
        print(f"\nEvaluating {name} | valid={int(ok.sum())}/{int(keep_mask.sum())}")
        if np.any(ok):
            ss = s[ok]
            print(f"  range=[{ss.min():.6f},{ss.max():.6f}] mean={ss.mean():.6f}±{ss.std():.6f}")
    pr = tail_metrics_at_k(
        scores, keep_mask, k_fracs,
        vulnerable=vulnerable,
        gold_scores=gold_scores,
        pool_member_scores=pool_member_scores,
        gold_topk=gold_topk,
    )
    if print_report:
        print(pr.to_string(index=False))
    return {"name": name, "scores": scores, "pr": pr}


def side_by_side_report(results: List[Dict[str, object]]) -> pd.DataFrame:
    """Per-target side-by-side per-k table for all methods.

    Columns per method `n`:
      n_precision, n_recall, n_tail_rho, n_jaccard
    """
    if not results:
        return pd.DataFrame()
    base = results[0]["pr"][["k", "k_samples"]].copy()
    out = base.copy()
    metric_cols = ["precision", "recall", "jaccard", "tail_rho", "gold_n"]
    for r in results:
        n = r["name"]
        keep = [c for c in metric_cols if c in r["pr"].columns]
        pr = r["pr"][["k"] + keep].rename(
            columns={c: f"{n}_{c}" for c in keep}
        )
        out = out.merge(pr, on="k", how="left")
    return out


def plot_overlapping_distributions(
    results: List[Dict[str, object]],
    vulnerable: np.ndarray,
    membership_mask: np.ndarray,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 8),
    bins: int = 60,
):
    n_methods = len(results)
    if n_methods == 0:
        return
    ncols = 2
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_methods > 1 else [axes]

    mem = membership_mask.astype(bool)
    vuln = vulnerable.astype(bool)

    for i, result in enumerate(results):
        name = result["name"]
        scores = result["scores"]
        ax = axes[i]

        scores_mem = scores[mem]
        vuln_mem = vuln[mem]
        valid = np.isfinite(scores_mem)
        scores_mem = scores_mem[valid]
        vuln_mem = vuln_mem[valid]

        if scores_mem.size == 0:
            ax.text(0.5, 0.5, "No valid scores", ha="center", va="center")
            continue

        scores_vuln = scores_mem[vuln_mem]
        scores_non_vuln = scores_mem[~vuln_mem]
        vmin, vmax = np.min(scores_mem), np.max(scores_mem)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            ax.text(0.5, 0.5, "Degenerate scores", ha="center", va="center")
            continue

        bin_edges = np.linspace(vmin, vmax, bins + 1)
        ax.hist(
            scores_non_vuln,
            bins=bin_edges,
            alpha=0.55,
            color="blue",
            label=f"Non-Vuln (n={len(scores_non_vuln)})",
            edgecolor="black",
            linewidth=0.25,
            density=True,
        )
        ax.hist(
            scores_vuln,
            bins=bin_edges,
            alpha=0.55,
            color="red",
            label=f"Vuln (n={len(scores_vuln)})",
            edgecolor="black",
            linewidth=0.25,
            density=True,
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25)

        if len(scores_vuln) > 0 and len(scores_non_vuln) > 0:
            ks_stat = stats.ks_2samp(scores_vuln, scores_non_vuln, alternative="two-sided").statistic
            ax.text(
                0.02,
                0.98,
                f"KS={ks_stat:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.6),
            )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


# =============================================================================
# Predicted classes on TRAIN under the original target model
# =============================================================================
@torch.no_grad()
def compute_predicted_classes_on_train(
    experiment_dir: str,
    target_model: int,
    config: dict,
    train_dataset_raw,
    num_classes: int,
    device: str,
    batch_size: int = 256,
    num_workers: int = 4,
    show_progress: bool = True,
) -> np.ndarray:
    from rare.utils.data import TransformSubset, build_transforms
    from rare.utils.model import get_model

    exp = Path(experiment_dir)
    model_dir = exp / f"model_{target_model}"
    model_path = model_dir / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    dev = torch.device(device)
    model = get_model(int(num_classes), **config.get("model", {}))
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(dev).eval()

    eval_tf = build_transforms(config, train=False)
    N = len(train_dataset_raw)
    all_idx = np.arange(N, dtype=np.int64)
    subset = TransformSubset(train_dataset_raw, all_idx, transform=eval_tf)
    indexed = IndexedDataset(subset)
    loader = DataLoader(
        indexed,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device == "cuda"),
        persistent_workers=False,
    )

    pred = np.full((N,), -1, dtype=np.int64)
    pb = ProgressBar("pred_class_train", total=len(loader), enabled=show_progress)
    for bi, (x, _, idx) in enumerate(loader, 1):
        x = x.to(dev, non_blocking=True)
        yhat = model(x).argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        pred[idx.numpy()] = yhat
        pb.update(bi)
    pb.close()

    if np.any(pred < 0):
        bad = int(np.sum(pred < 0))
        raise RuntimeError(f"Failed to compute predicted classes for {bad} train samples.")
    return pred


# =============================================================================
# LT-IQR
# =============================================================================
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
# Locked RARE: z_M(LT-BTM-R) + z_M(LT-IQR)
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
    """Load trajectories and compute locked RARE = z_M(LT-BTM-R) + z_M(LT-IQR).

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


# =============================================================================
# Simple distance-based vulnerability scores.
# Per-sample distance between the target trajectory and each reference trajectory,
# flattened across (E, F_slot) and averaged over R references.
#
# Polarity intuition:
#   - target members: target=member-shape, ref=non-member-shape (flipped) → HIGH distance
#   - unflipped target non-members: both non-member-shape → LOW distance
#   - flipped target non-members: target=non-member, ref=member-shape → HIGH distance
# So HIGH score = member. Useful as a no-training baseline; flipped non-members
# will inflate AUC noise but member-rank metrics ignore them.
# =============================================================================
def per_sample_target_ref_distance(
    target_NEF: np.ndarray,
    ref_list_NEF: List[np.ndarray],
    metric: str = "euclidean",
    z_norm_channels: bool = True,
) -> np.ndarray:
    """Mean over R references of dist(target_traj_flat, ref_r_traj_flat).

    Parameters
    ----------
    target_NEF : (N, E, F_slot) target trajectory.
    ref_list_NEF : list of R arrays each (N, E, F_slot).
    metric : "euclidean" or "cosine".
    z_norm_channels : if True, robust-z each channel across (N×E) before
        flattening, so loss and margin contribute on comparable scales.

    Returns
    -------
    scores : (N,) fp32. High = target member.
    """
    R = len(ref_list_NEF)
    if R == 0:
        return np.full(target_NEF.shape[0], np.nan, dtype=np.float32)

    for r, ref in enumerate(ref_list_NEF):
        if ref.shape != target_NEF.shape:
            raise ValueError(
                f"ref_list_NEF[{r}].shape={ref.shape} != target {target_NEF.shape}"
            )

    N, E, F = target_NEF.shape
    t = target_NEF.astype(np.float32, copy=False)
    cs = [ref.astype(np.float32, copy=False) for ref in ref_list_NEF]

    if z_norm_channels:
        union = np.concatenate(
            [t.reshape(-1, F)] + [c.reshape(-1, F) for c in cs], axis=0
        ).astype(np.float32)
        med = np.median(union, axis=0, keepdims=True).astype(np.float32)
        mad = np.median(np.abs(union - med), axis=0, keepdims=True).astype(np.float32)
        scale = np.maximum(1.4826 * mad, np.float32(1e-8))

        def _apply(arr: np.ndarray) -> np.ndarray:
            out = (arr - med.reshape(1, 1, F)) / scale.reshape(1, 1, F)
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        t = _apply(t)
        cs = [_apply(c) for c in cs]

    t_flat = t.reshape(N, -1)                                      # (N, E*F)
    scores = np.zeros(N, dtype=np.float32)
    metric_l = str(metric).lower()
    for ref in cs:
        ref_flat = ref.reshape(N, -1)
        if metric_l == "euclidean":
            diff = t_flat - ref_flat
            d = np.sqrt(np.sum(diff * diff, axis=1))
        elif metric_l == "cosine":
            num = np.sum(t_flat * ref_flat, axis=1)
            den = (
                np.linalg.norm(t_flat, axis=1)
                * np.linalg.norm(ref_flat, axis=1)
                + 1e-12
            )
            d = 1.0 - num / den
        else:
            raise ValueError(f"metric must be euclidean|cosine, got {metric!r}")
        scores += d.astype(np.float32, copy=False)
    return (scores / float(R)).astype(np.float32, copy=False)


# =============================================================================
# Config-override helpers (YAML-aware; supports lists/bools/numbers/null)
# =============================================================================
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


def validate_required(config: Dict[str, Any], keys: List[str]) -> None:
    def has_keypath(d: Dict[str, Any], path: str) -> bool:
        cur = d
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return False
            cur = cur[p]
        return True

    missing = [k for k in keys if not has_keypath(config, k)]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
