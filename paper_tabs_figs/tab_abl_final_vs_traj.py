"""Reference-statistic and component ablation (tab:abl-final-vs-traj).

Given the same single (M=1) reference model, is difficulty better read from its
loss trajectory than from its final checkpoint? Per dataset (CIFAR10/100,
CINIC10) and target (10), eight reference-side statistics are formed from the
reference loss trajectory:
  position-dependent  REF-FINAL, CALIB-FINAL (= ref - target final loss), REF-LAST
  order-invariant     REF-MIN, REF-MAX, REF-MEAN, LT-TOP-R, LT-BTM-R (q=50%)
Each is evaluated (a) alone and (b) combined as z_T(stat) + z_T(LT-IQR), the
RARE form. P@{1,3,5}% (ceil budget, uncapped) under LiRA-256 gold at
FPR<=1e-4, mean+/-std over the 10 targets, plus the cross-dataset average.

Outputs: tables/tab_abl_final_vs_trajectory.{csv,tex}. CPU-only; nothing is
retrained.

Usage:
  python paper_tabs_figs/tab_abl_final_vs_traj.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rare.analysis.utils import (  # noqa: E402
    score_avg_min_q_reference_losses, score_avg_max_q_reference_losses,
    score_ltiqr_trajectory, _zscore_ignore_nan, tail_metrics_at_k)
from paper_tabs_figs._tex import ADJUSTBOX_OPEN, pm_scriptsize  # noqa: E402

import paper_tabs_figs._scores as mt  # noqa: E402

BENCHMARKS = {
    "CIFAR10": ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140",
    "CIFAR100": ROOT / "experiments/cifar100/wrn28-2/2025-12-25_0121",
    "CINIC10": ROOT / "experiments/cinic10/wrn28-2/2026-05-11_0215",
}
TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
K_FRACS = [0.01, 0.03, 0.05]
K_LABELS = ["P@1%", "P@3%", "P@5%"]
FPR = 1e-4
Q = 0.50
_REF = re.compile(r"^ref_losses_E(\d+)_(.+)_ref0\.npy$")

# Three blocks: the target-only baseline, the eight reference-side statistics
# alone, then each combined with LT-IQR as z_T(.)+z_T(LT-IQR).
#   position-dependent : REF-FINAL, CALIB-FINAL, REF-LAST
#   order-invariant    : REF-MAX, REF-MIN, REF-MEAN, LT-TOP-R, LT-BTM-R
# REF-MIN / LT-BTM-R / REF-MEAN are the B_q in {1, E/2, E} members of the
# lower-tail family; REF-LAST averages the same count as LT-BTM-R but selects
# by position rather than by loss value.
BASELINE_ROW = "LT-IQR"
ALONE_ROWS = ["REF-FINAL", "CALIB-FINAL", "REF-LAST", "REF-MAX",
              "REF-MIN ($B_q{=}1$)", "REF-MEAN ($B_q{=}E$)",
              "LT-TOP-R ($q{=}50\\%$)", "LT-BTM-R ($q{=}50\\%$)"]
COMBINED_ROWS = ["LT-IQR + REF-FINAL", "LT-IQR + CALIB-FINAL",
                 "LT-IQR + REF-LAST", "LT-IQR + REF-MAX",
                 "LT-IQR + REF-MIN", "LT-IQR + REF-MEAN",
                 "LT-IQR + LT-TOP-R", "\\rare{} (LT-IQR + LT-BTM-R)"]
ROW_ORDER = [BASELINE_ROW] + ALONE_ROWS + COMBINED_ROWS
# \midrule is emitted after each of these rows.
MIDRULE_AFTER = {BASELINE_ROW, ALONE_ROWS[-1]}
def _load_traj(exp: Path, t: int):
    """(reference traj (N,E), target traj (N,E), members mask)."""
    td = exp / f"model_{t}"
    tgt = np.load(td / "per_epoch_losses.npy").astype(np.float32).T   # (N, E)
    ref_p = next(p for p in sorted(td.glob("ref_losses_E*_*_ref0.npy")) if _REF.match(p.name))
    ref = np.load(ref_p).astype(np.float32)                          # (N, E)
    keep = np.load(exp / "keep_indices.npy").astype(bool)[t]
    assert ref.shape == tgt.shape, (ref.shape, tgt.shape)
    return ref, tgt, keep


def _stats(ref, tgt):
    ref_final = ref[:, -1].astype(np.float32)
    tgt_final = tgt[:, -1].astype(np.float32)
    ref_mean = score_avg_min_q_reference_losses([ref], q=1.0)
    lt_btm_r = score_avg_min_q_reference_losses([ref], q=Q)
    lt_top_r = score_avg_max_q_reference_losses([ref], q=Q)
    lt_iqr = score_ltiqr_trajectory(tgt)
    # Watson-style difficulty calibration at the final epoch: ref - target loss.
    calib_final = (ref_final - tgt_final).astype(np.float32)
    # REF-LAST: mean over the final half of training, by position. Same
    # checkpoint count as LT-BTM-R (E even), selected by epoch not by value.
    start = ref.shape[1] // 2
    ref_last = ref[:, start:].mean(axis=1).astype(np.float32)
    assert ref.shape[1] - start == max(1, int(round(Q * ref.shape[1]))), \
        "REF-LAST and LT-BTM-R must average the same number of checkpoints"
    return {"REF-FINAL": ref_final, "TGT-FINAL": tgt_final,
            "CALIB-FINAL": calib_final, "REF-LAST": ref_last,
            "REF-MEAN": ref_mean, "REF-MIN": ref.min(axis=1).astype(np.float32),
            "REF-MAX": ref.max(axis=1).astype(np.float32),
            "LT-BTM-R": lt_btm_r, "LT-TOP-R": lt_top_r, "LT-IQR": lt_iqr}


def _masked(x, members):
    out = np.full(x.shape, np.nan, dtype=np.float32)
    out[members] = x[members]
    return out


# Row label -> key in the _stats dict. Labels may carry a parenthetical
# annotation (e.g. "REF-MIN ($B_q{=}1$)"), so match on the leading token.
_STAT_OF = ["REF-FINAL", "CALIB-FINAL", "REF-LAST", "REF-MEAN", "REF-MIN",
            "REF-MAX", "LT-BTM-R", "LT-TOP-R", "LT-IQR"]


def _stat_key(label: str) -> str:
    for key in _STAT_OF:
        if label.startswith(key):
            return key
    raise ValueError(f"no statistic matches row label {label!r}")


def _row_score(label, s, members):
    # Combined rows: z_T(stat) + z_T(LT-IQR).
    if label.startswith("\\rare"):
        return (_zscore_ignore_nan(s["LT-BTM-R"], members)
                + _zscore_ignore_nan(s["LT-IQR"], members))
    if label.startswith("LT-IQR + "):
        key = _stat_key(label[len("LT-IQR + "):])
        return (_zscore_ignore_nan(s[key], members)
                + _zscore_ignore_nan(s["LT-IQR"], members))
    # Standalone rows: rank by the raw statistic, no z-scoring.
    if label not in ROW_ORDER:
        raise ValueError(label)
    return _masked(s[_stat_key(label)], members)


def evaluate():
    """results[ds][label] = (T, len(K)) P@k per target; tgt_final[ds] = member final losses."""
    results = {ds: {lab: [] for lab in ROW_ORDER} for ds in BENCHMARKS}
    tgt_final = {ds: [] for ds in BENCHMARKS}
    for ds, exp in BENCHMARKS.items():
        for t in TARGETS:
            ref, tgt, members = _load_traj(exp, t)
            s = _stats(ref, tgt)
            gold, mem_mask, vulnerable = mt.load_lira_gold(exp, t, FPR)
            assert np.array_equal(mem_mask, members)
            for lab in ROW_ORDER:
                pr = tail_metrics_at_k(_row_score(lab, s, members), members, K_FRACS,
                                       vulnerable=vulnerable, gold_scores=gold)
                results[ds][lab].append([float(pr.iloc[i]["precision"])
                                         for i in range(len(K_FRACS))])
            tgt_final[ds].append(s["TGT-FINAL"][members])
    for ds in results:
        for lab in results[ds]:
            results[ds][lab] = np.asarray(results[ds][lab])  # (T, K)
        tgt_final[ds] = np.concatenate(tgt_final[ds])
    return results, tgt_final


def write_outputs(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_names = list(BENCHMARKS)
    nk = len(K_FRACS)

    # means[label] -> (ds_means (3,K), per-cell mean/std for table)
    cell_mean = {lab: np.zeros((len(ds_names) + 1, nk)) for lab in ROW_ORDER}
    cell_std = {lab: np.zeros((len(ds_names) + 1, nk)) for lab in ROW_ORDER}
    for lab in ROW_ORDER:
        ds_means = []
        for di, ds in enumerate(ds_names):
            arr = results[ds][lab]                      # (T, K)
            cell_mean[lab][di] = arr.mean(0); cell_std[lab][di] = arr.std(0, ddof=1)
            ds_means.append(arr.mean(0))
        ds_means = np.asarray(ds_means)
        cell_mean[lab][-1] = ds_means.mean(0); cell_std[lab][-1] = ds_means.std(0, ddof=1)

    # ---- CSV (long) ----
    import csv as _csv
    with open(out_dir / "tab_abl_final_vs_trajectory.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["method", "dataset", "k", "mean", "std"])
        blocks = ds_names + ["Avg"]
        for lab in ROW_ORDER:
            for blk, ds in enumerate(blocks):
                for j, k in enumerate(K_LABELS):
                    w.writerow([lab.replace("\\rare{}", "RARE").replace("\\%", "%"),
                                ds, k, f"{cell_mean[lab][blk, j]:.4f}",
                                f"{cell_std[lab][blk, j]:.4f}"])

    # ---- LaTeX (as in the paper: no best/second marks) ----
    lines = [
        "% Auto-generated by paper_tabs_figs/tab_abl_final_vs_traj.py -- do not edit.",
        r"""\begin{table*}[t]
\centering
\caption{Reference-statistic and component ablation. Each reference-side
statistic is evaluated alone and combined with LT-IQR via the
standardized sum $z_T(\mathrm{LT\text{-}IQR})+z_T(\cdot)$. 
Results are P@$k$ under LiRA-256 gold at $\mathrm{FPR}\leq 10^{-4}$.
Per-dataset entries are mean$\pm$std over 10 targets; avg.\ entries are
the mean and standard deviation across the three datasets.}
\label{tab:abl-final-vs-traj}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}""",
        ADJUSTBOX_OPEN,
        "\\begin{tabular}{l ccc ccc ccc ccc}", "\\toprule",
        "\\multirow{2}{*}{\\textbf{Score}}"
        + "".join(f"  & \\multicolumn{{3}}{{c}}{{\\textbf{{{n}}}}}"
                  for n in ds_names + ["Avg."]) + " \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}",
        " " + " ".join("& \\textbf{P@1\\%} & \\textbf{P@3\\%} & \\textbf{P@5\\%}"
                       for _ in range(4)) + " \\\\",
        "\\midrule",
    ]
    for lab in ROW_ORDER:
        block = [lab]
        for blk in range(len(ds_names) + 1):
            block.append("& " + " & ".join(
                pm_scriptsize(cell_mean[lab][blk, j], cell_std[lab][blk, j])
                for j in range(nk)))
        lines.append("\n".join(block) + " \\\\")
        if lab in MIDRULE_AFTER:   # block separators
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    (out_dir / "tab_abl_final_vs_trajectory.tex").write_text("\n".join(lines) + "\n")
    return cell_mean, cell_std


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "paper_tabs_figs/tables")
    args = ap.parse_args()

    results, tgt_final = evaluate()
    write_outputs(results, args.out_dir)

    # Supports the text: CALIB-FINAL subtracts a member loss that is ~0.
    print("Target final-epoch loss over members (memorized => ~0):")
    for ds, v in tgt_final.items():
        print(f"  {ds:<10} mean={v.mean():.4f}  p99={np.percentile(v, 99):.4f}")

    print("\nWrote:")
    for fn in ("tab_abl_final_vs_trajectory.csv", "tab_abl_final_vs_trajectory.tex"):
        print(f"  {args.out_dir / fn}")


if __name__ == "__main__":
    main()
