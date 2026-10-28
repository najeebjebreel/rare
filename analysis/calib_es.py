"""Early-stopped difficulty calibration (text of Section "abl-final-vs-traj").

Watson et al. (ICLR 2022) calibrate the loss attack as l_R - l_T with reference
models trained until their validation accuracy peaks. Our CALIB-FINAL row reads
the single (M=1) reference at its final epoch E. This script reads it instead at
the epoch of peak reference accuracy, per target:

  e*          = argmax_e accuracy of the reference on the target members
                (held out for the reference, which is trained on the target's
                non-members); earliest epoch on ties
  CALIB-ES    = l_R[i, e*] - l_T[i, E]

and reports the range of e* and the change in P@{1,3,5}% (LiRA-256 gold at
FPR <= 1e-4) relative to CALIB-FINAL. Only existing artifacts are used:
the per-epoch reference loss and predicted-class traces, and the dataset labels.

Usage:
  python analysis/calib_es.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rare.analysis.utils import tail_metrics_at_k  # noqa: E402

BENCHMARKS = {
    "CIFAR10": ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140",
    "CIFAR100": ROOT / "experiments/cifar100/wrn28-2/2025-12-25_0121",
    "CINIC10": ROOT / "experiments/cinic10/wrn28-2/2026-05-11_0215",
}
TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
K_FRACS = [0.01, 0.03, 0.05]
FPR = 1e-4
_REF = re.compile(r"^ref_losses_E(\d+)_(.+)_ref0\.npy$")


def _labels(ds: str, exp: Path) -> np.ndarray:
    """Class labels in the experiment's training-pool order."""
    data = ROOT / "data"
    if ds == "CIFAR10":
        ys = []
        for b in range(1, 6):
            with open(data / "cifar-10-batches-py" / f"data_batch_{b}", "rb") as f:
                ys += pickle.load(f, encoding="bytes")[b"labels"]
        return np.asarray(ys, dtype=np.int64)
    if ds == "CIFAR100":
        with open(data / "cifar-100-python" / "train", "rb") as f:
            return np.asarray(pickle.load(f, encoding="bytes")[b"fine_labels"], dtype=np.int64)
    import lmdb  # CINIC10: LMDB records are (image bytes, label), subsampled
    sub = np.load(exp / "subsample_indices.npy").astype(np.int64)
    env = lmdb.open(str(data / "cinic10" / "train.lmdb"), readonly=True, lock=False,
                    subdir=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        y = [pickle.loads(txn.get(str(int(i)).encode()))[1] for i in sub]
    env.close()
    return np.asarray(y, dtype=np.int64)


def _precision(score, members, vulnerable, gold):
    out = np.full(score.shape, np.nan, dtype=np.float32)
    out[members] = score[members]
    pr = tail_metrics_at_k(out, members, K_FRACS, vulnerable=vulnerable, gold_scores=gold)
    return [float(pr.iloc[i]["precision"]) for i in range(len(K_FRACS))]


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    epochs, delta = {}, {}
    for ds, exp in BENCHMARKS.items():
        y = _labels(ds, exp)
        keep = np.load(exp / "keep_indices.npy").astype(bool)
        epochs[ds], es, fin = [], [], []
        for t in TARGETS:
            td = exp / f"model_{t}"
            ref_p = next(p for p in sorted(td.glob("ref_losses_E*_*_ref0.npy")) if _REF.match(p.name))
            E, tag = _REF.match(ref_p.name).groups()
            ref = np.load(ref_p).astype(np.float32)                        # (N, E)
            pred = np.load(td / f"ref_pred_class_E{E}_{tag}_ref0.npy")      # (N, E)
            tgt_final = np.load(td / "per_epoch_losses.npy").astype(np.float32)[-1]
            members = keep[t]
            acc = (pred[members] == y[members, None]).mean(axis=0)
            e_star = int(np.flatnonzero(acc == acc.max())[0])
            epochs[ds].append(e_star + 1)

            ar = td / "attack_results"
            gold = np.load(ar / "lira_refs_256_scores.npy").astype(np.float32)
            thr = np.load(ar / "lira_refs_256_thresholds_by_fpr.npy", allow_pickle=True).item()[FPR]
            vulnerable = (gold >= float(thr)) & members
            es.append(_precision(ref[:, e_star] - tgt_final, members, vulnerable, gold))
            fin.append(_precision(ref[:, -1] - tgt_final, members, vulnerable, gold))
        delta[ds] = np.mean(es, axis=0) - np.mean(fin, axis=0)
        print(f"  {ds} done", file=sys.stderr, flush=True)

    all_e = np.concatenate([epochs[ds] for ds in BENCHMARKS])
    print(f"Peak-accuracy epoch of the reference: {all_e.min()}-{all_e.max()} "
          f"(of {E}); per dataset: "
          + ", ".join(f"{ds} {min(v)}-{max(v)}" for ds, v in epochs.items()))
    print("Change in mean P@1/3/5% from CALIB-FINAL to CALIB-ES:")
    for ds, d in delta.items():
        print(f"  {ds:<9} " + " / ".join(f"{x:+.4f}" for x in d))
    avg = np.mean([delta[ds] for ds in BENCHMARKS], axis=0)
    print(f"  {'Avg':<9} " + " / ".join(f"{x:+.4f}" for x in avg))
    print(f"Max |change| over datasets and depths: "
          f"{max(np.abs(d).max() for d in delta.values()):.4f}")


if __name__ == "__main__":
    main()
