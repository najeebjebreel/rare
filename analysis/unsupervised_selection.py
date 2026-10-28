"""Unsupervised vulnerable-set selection (Section "Unsupervised vulnerable-set selection").

For each target, two-cluster K-means is applied to the member-standardized
LT-IQR and LT-BTM-R (q=50%, M=1) component scores, and the cluster with the
higher mean RARE score (their sum) is selected. The rule uses no gold labels
and no threshold calibration, so the same selected set S is evaluated against
LiRA-256 gold sets at three operating points: FPR = 0, FPR <= 1e-4 and
FPR <= 1e-3.

Prints the numbers quoted in the paper:
  * |S| / n_mem per dataset, and |V| / n_mem at FPR <= 1e-4;
  * recall and precision of S against each gold set (range across datasets);
  * recall against the FPR <= 1e-4 gold set per dataset.

Usage:
  python analysis/unsupervised_selection.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# sklearn's KMeans reduction order depends on the OpenMP thread count, so
# random_state alone does not make it reproducible across processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rare.analysis.utils import (  # noqa: E402
    _zscore_ignore_nan, score_avg_min_q_reference_losses)

BENCHMARKS = {
    "CIFAR10": ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140",
    "CIFAR100": ROOT / "experiments/cifar100/wrn28-2/2025-12-25_0121",
    "CINIC10": ROOT / "experiments/cinic10/wrn28-2/2026-05-11_0215",
}
TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
FPRS = [0.0, 1e-4, 1e-3]
Q = 0.50


def load_gold(exp: Path, t: int, fpr: float):
    """(members, vulnerable) from the saved LiRA-256 scores and FPR thresholds."""
    ar = exp / f"model_{t}" / "attack_results"
    scores = np.load(ar / "lira_refs_256_scores.npy").astype(np.float32)
    members = np.load(ar / "lira_refs_256_labels.npy").astype(bool)
    thr = np.load(ar / "lira_refs_256_thresholds_by_fpr.npy", allow_pickle=True).item()
    return members, (scores >= float(thr[float(fpr)])) & members


def kmeans_select(exp: Path, t: int, members: np.ndarray) -> np.ndarray:
    """Boolean mask of the selected members (higher-RARE K-means cluster)."""
    td = exp / f"model_{t}"
    lt_iqr = np.load(td / "lt_iqr_scores.npy").astype(np.float32)
    ref = np.load(sorted(td.glob("ref_losses_E*_*_ref0.npy"))[0]).astype(np.float32)
    lt_btm = score_avg_min_q_reference_losses([ref], q=Q)
    z_iqr, z_btm = _zscore_ignore_nan(lt_iqr, members), _zscore_ignore_nan(lt_btm, members)
    ok = members & np.isfinite(z_iqr) & np.isfinite(z_btm)
    idx = np.where(ok)[0]
    labels = KMeans(n_clusters=2, random_state=0, n_init=50).fit_predict(
        np.column_stack([z_iqr, z_btm])[idx])
    rare = (z_iqr + z_btm)[idx]
    chosen = int(np.argmax([rare[labels == c].mean() for c in (0, 1)]))
    selected = np.zeros(members.shape, dtype=bool)
    selected[idx[labels == chosen]] = True
    return selected


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    # stats[ds][fpr] = list of per-target dicts
    stats = {ds: {f: [] for f in FPRS} for ds in BENCHMARKS}
    for ds, exp in BENCHMARKS.items():
        for t in TARGETS:
            members, _ = load_gold(exp, t, FPRS[0])
            S = kmeans_select(exp, t, members)
            for f in FPRS:
                m_f, V = load_gold(exp, t, f)
                assert np.array_equal(m_f, members)
                tp = int((S & V).sum())
                stats[ds][f].append(dict(n_mem=int(members.sum()), S=int(S.sum()),
                                         V=int(V.sum()), recall=tp / V.sum(),
                                         precision=tp / S.sum()))
        print(f"  {ds} done", file=sys.stderr, flush=True)

    def mean(ds, f, key):
        return float(np.mean([r[key] for r in stats[ds][f]]))

    print("Selected fraction |S|/n_mem and vulnerable fraction |V|/n_mem (FPR<=1e-4):")
    for ds in BENCHMARKS:
        frac_s = np.mean([r["S"] / r["n_mem"] for r in stats[ds][1e-4]]) * 100
        frac_v = np.mean([r["V"] / r["n_mem"] for r in stats[ds][1e-4]]) * 100
        print(f"  {ds:<9} selected {frac_s:.1f}%   vulnerable {frac_v:.1f}%")

    print("\nRecall / precision of S against LiRA-256 gold (range over dataset means):")
    for f in FPRS:
        rec = [mean(ds, f, "recall") for ds in BENCHMARKS]
        pre = [mean(ds, f, "precision") for ds in BENCHMARKS]
        print(f"  FPR {'=' if f == 0 else '<='} {f:g}: recall {min(rec):.2f}-{max(rec):.2f}"
              f"  precision {min(pre):.2f}-{max(pre):.2f}")

    print("\nRecall against the FPR<=1e-4 gold set, per dataset:")
    print("  " + ", ".join(f"{ds} {mean(ds, 1e-4, 'recall'):.2f}" for ds in BENCHMARKS))


if __name__ == "__main__":
    main()
