"""Ablation: reference final-epoch loss vs RARE's trajectory statistic (LT-BTM-R).

Question: does RARE's reference-side bottom-q% trajectory mean (q=50%) beat the
standard difficulty-calibration choice of the reference model's FINAL-epoch loss?

Per dataset (CIFAR-10/100/CINIC-10) and per target (10), from the existing M=1
reference loss trajectories, we form three reference-side statistics per member:
  REF-FINAL    reference loss at the last epoch         traj[:, -1]
  REF-MEAN     mean over all epochs (q=100%)            avg_min_q(traj, q=1.0)
  LT-BTM-R     bottom-50% epoch mean (locked, q=50%)    avg_min_q(traj, q=0.5)
and evaluate each (a) alone and (b) combined as z_M(stat)+z_M(LT-IQR) (the locked
RARE form). Capped P@{1,3,5,10}% under LiRA-256 gold @FPR<=1e-4, mean+/-std over
the 10 targets, plus cross-dataset average.

Outputs: tables/tab_abl_final_vs_trajectory.{csv,tex}. CPU-only; nothing is
retrained.

Usage:
  python paper_tabs_figs/tab_abl_final_vs_traj.py
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rare.analysis.utils import (  # noqa: E402
    score_avg_min_q_reference_losses, score_avg_max_q_reference_losses,
    score_ltiqr_trajectory, _zscore_ignore_nan, tail_metrics_at_k)

_spec = importlib.util.spec_from_file_location("mt", ROOT / "scripts" / "main_table.py")
mt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mt)

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
# alone, then each combined with LT-IQR as z_M(.)+z_M(LT-IQR).
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
    # Combined rows: z_M(stat) + z_M(LT-IQR).
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
    # results[ds][label] = (T, len(K)) capped precision per target.
    results = {ds: {lab: [] for lab in ROW_ORDER} for ds in BENCHMARKS}
    diag = {"loss_corr": [],                               # REF-FINAL vs LOSS
            "cf_reff_spearman": {ds: [] for ds in BENCHMARKS},  # CALIB vs REF-FINAL
            "tgt_final": {ds: [] for ds in BENCHMARKS}}     # target final loss / member
    for ds, exp in BENCHMARKS.items():
        for t in TARGETS:
            ref, tgt, members = _load_traj(exp, t)
            s = _stats(ref, tgt)
            gold, mem_mask, vulnerable = mt._load_lira_gold(exp, t, FPR)
            assert np.array_equal(mem_mask, members)
            for lab in ROW_ORDER:
                pr = tail_metrics_at_k(_row_score(lab, s, members), members, K_FRACS,
                                       vulnerable=vulnerable, gold_scores=gold,
                                       cap_k_at_gold=False, k_rounding="ceil")
                results[ds][lab].append([float(pr.iloc[i]["precision"])
                                         for i in range(len(K_FRACS))])
            # REF-FINAL (reference) vs the LOSS baseline (target final loss).
            loss_p = exp / f"model_{t}" / "attack_results" / "loss_scores.npy"
            if loss_p.exists():
                loss = np.load(loss_p).astype(np.float32)
                a, b = s["REF-FINAL"][members], loss[members]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() > 2:
                    diag["loss_corr"].append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
            # CALIB-FINAL vs REF-FINAL Spearman over members (should be ~1).
            cf, rf = s["CALIB-FINAL"][members], s["REF-FINAL"][members]
            diag["cf_reff_spearman"][ds].append(float(spearmanr(cf, rf).statistic))
            diag["tgt_final"][ds].append(s["TGT-FINAL"][members])
    for ds in results:
        for lab in results[ds]:
            results[ds][lab] = np.asarray(results[ds][lab])  # (T, K)
        diag["tgt_final"][ds] = np.concatenate(diag["tgt_final"][ds])
    return results, diag


def _cell(mean, std, bold):
    body = f"{mean:.2f}"
    if bold:
        body = f"\\textbf{{{body}}}"
    return f"{body}{{\\scriptsize$\\pm${std:.2f}}}"


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

    # Bold the best per (dataset-block, k) column, including ties, decided at
    # full precision; displayed values are rounded to two decimals.
    best = {}
    for blk in range(len(ds_names) + 1):
        for j in range(nk):
            top = max(cell_mean[L][blk, j] for L in ROW_ORDER)
            best[(blk, j)] = {L for L in ROW_ORDER
                              if cell_mean[L][blk, j] >= top - 1e-12}

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

    # ---- LaTeX (matches the component ablation table format) ----
    lines = [
        "% Auto-generated by paper_tabs_figs/tab_abl_final_vs_traj.py -- do not edit.",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Reference final-epoch loss vs.\\ trajectory statistics for the"
        " reference-side signal. Each statistic alone and combined with LT-IQR as"
        " $z_M(\\cdot)+z_M(\\mathrm{LT\\text{-}IQR})$. LT-TOP-R is the top-$q$"
        " mirror of LT-BTM-R. P@$k$ under LiRA-256 gold at FPR$\\le10^{-4}$,"
        " mean$\\pm$std over 10 targets.}",
        "\\label{tab:abl-final-vs-traj}",
        "\\setlength{\\tabcolsep}{3pt}", "\\renewcommand{\\arraystretch}{1.05}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l ccc ccc ccc ccc}", "\\toprule",
        "\\multirow{2}{*}{\\textbf{Score}}"
        + "".join(f"  & \\multicolumn{{3}}{{c}}{{\\textbf{{{n}}}}}"
                  for n in ds_names + ["Avg."]) + " \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}",
        " " + " ".join("& \\textbf{P@1\\%} & \\textbf{P@3\\%} & \\textbf{P@5\\%}"
                       for _ in range(4)) + " \\\\",
        "\\midrule",
    ]
    for ri, lab in enumerate(ROW_ORDER):
        cells = [lab]
        for blk in range(len(ds_names) + 1):
            for j in range(nk):
                cells.append(_cell(cell_mean[lab][blk, j], cell_std[lab][blk, j],
                                   lab in best[(blk, j)]))
        block = [cells[0]]
        for blk in range(len(ds_names) + 1):
            block.append("& " + " & ".join(cells[1 + blk * nk:1 + (blk + 1) * nk]))
        lines.append("\n".join(block) + " \\\\")
        if lab in MIDRULE_AFTER:   # block separators
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    (out_dir / "tab_abl_final_vs_trajectory.tex").write_text("\n".join(lines) + "\n")
    return cell_mean, cell_std, best


def _format_row(lab, cell_mean, cell_std, best, n_ds, nk):
    cells = [lab]
    for blk in range(n_ds + 1):
        for j in range(nk):
            cells.append(_cell(cell_mean[lab][blk, j], cell_std[lab][blk, j],
                               lab in best[(blk, j)]))
    block = [cells[0]]
    for blk in range(n_ds + 1):
        block.append("& " + " & ".join(cells[1 + blk * nk:1 + (blk + 1) * nk]))
    return "\n".join(block) + " \\\\"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "paper_tabs_figs/tables")
    args = ap.parse_args()

    results, diag = evaluate()
    cell_mean, cell_std, best = write_outputs(results, args.out_dir)
    ds_only = list(BENCHMARKS)
    nk = len(K_FRACS)

    print("\n=== Sanity: REF-FINAL (reference final loss) vs LOSS baseline "
          "(target final loss) ===")
    print(f"  Pearson r over members, mean across targets/datasets = "
          f"{np.nanmean(diag['loss_corr']):.3f}  (clearly < 1 => different signals)")
    print(f"  REF-FINAL-alone avg P@5% = {cell_mean['REF-FINAL'][-1, 2]:.3f} vs "
          f"LOSS baseline ~0.02 => REF-FINAL is a real difficulty signal.")

    print("\n=== Sanity: CALIB-FINAL = ref_final - target_final ===")
    print("  target_traj[:,-1] over members (memorized => ~0):")
    for ds in ds_only:
        v = diag["tgt_final"][ds]
        print(f"    {ds:<10} mean={v.mean():.4f}  p99={np.percentile(v, 99):.4f}")
    print("  Spearman(CALIB-FINAL, REF-FINAL) over members (expect ~1):")
    for ds in ds_only:
        sp = np.array(diag["cf_reff_spearman"][ds])
        print(f"    {ds:<10} {sp.mean():.4f} ± {sp.std(ddof=1):.4f}")

    print("\n=== Check: CALIB-FINAL rows within +/-1pp of REF-FINAL rows ===")
    pairs = [("CALIB-FINAL", "REF-FINAL"),
             ("LT-IQR + CALIB-FINAL", "LT-IQR + REF-FINAL")]
    blocks = ds_only + ["Avg"]
    worst = 0.0
    for ca, rf in pairs:
        for blk, ds in enumerate(blocks):
            d = (cell_mean[ca][blk] - cell_mean[rf][blk]) * 100
            worst = max(worst, float(np.abs(d).max()))
            flags = [f"{K_LABELS[j]}:{d[j]:+.1f}" for j in range(nk) if abs(d[j]) > 1.0]
            if flags:
                print(f"  [>1pp] {ca} vs {rf} @ {ds}: " + ", ".join(flags))
    print(f"  max |Δ| across all CALIB-vs-REF-FINAL cells = {worst:.2f} pp"
          + ("  (all within +/-1pp)" if worst <= 1.0 else "  (SOME exceed 1pp, flagged above)"))

    print("\n=== Drop-in LaTeX rows for tab:abl-final-vs-traj ===")
    print("% -- place after the REF-FINAL row (alone block):")
    print(_format_row("CALIB-FINAL", cell_mean, cell_std, best, len(ds_only), nk))
    print("% -- place after the LT-IQR + REF-FINAL row (combined block):")
    print(_format_row("LT-IQR + CALIB-FINAL", cell_mean, cell_std, best, len(ds_only), nk))

    print("\nWrote:")
    for fn in ("tab_abl_final_vs_trajectory.csv", "tab_abl_final_vs_trajectory.tex"):
        print(f"  {args.out_dir / fn}")


if __name__ == "__main__":
    main()
