"""Generate the budget-equivalence LaTeX table (tab:crossover).

CIFAR10 under LiRA-256 gold at FPR<=1e-4. Two blocks:

  upper  a standalone difficulty ranker that scores each member by its MEAN
         FINAL-EPOCH loss over M complement-trained reference models, swept
         over M. These are the per-target references from
         scripts/rare_train_references.py (ref0..ref7), each trained on the
         target's non-members, so a single reference is OUT for every member
         at once and M references == M models trained. That keeps the sweep
         axis a genuine training budget, directly comparable to \\rare{}'s M.

         (A per-sample OUT draw from the 266-model shadow pool was evaluated
         and rejected: there, M is an OUT count per sample, and covering
         every sample at M=3 already costs 26 trained models, so the axis is
         no longer a budget.)
  lower  the \\rare{} components and their combination, with their reference
         budgets: LT-IQR (M=0), LT-BTM-R (M=1), \\rare{} (M=1).

Budget equivalents are reported only for LT-IQR and LT-BTM-R. \\rare{} lands
past M=8, where the curve gains ~0.004 per reference against an M->inf
ceiling only ~0.02 above it, so its interpolated equivalent is too
ill-conditioned to quote.

Precision@k uses a retrieval budget of ceil(k% * n_members), untruncated.

Interpolating the upper block's P@5% curve against each lower-block score
gives that component a *budget equivalent*: the number of reference models a
plain final-loss ranker would need to match it. Linear interpolation between
adjacent grid points; reported alongside the table.

Outputs: tables/tab_crossover.{csv,tex}. CPU-only; nothing is retrained.

Usage:
  python paper_tabs_figs/tab_crossover.py
"""

from __future__ import annotations

import argparse
import csv as _csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rare.analysis.utils import (  # noqa: E402
    load_rare_locked_scores, score_avg_min_q_reference_losses,
    score_ltiqr_trajectory, tail_metrics_at_k)
from paper_tabs_figs._tex import ADJUSTBOX_OPEN, pm_scriptstyle  # noqa: E402

EXP = ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140"
TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
K_FRACS = [0.01, 0.03, 0.05]
K_LABELS = ["P@1%", "P@3%", "P@5%"]
FPR = 1e-4
Q = 0.50
M_GRID = [1, 2, 3, 4, 5, 6, 8]
EQUIV_K = 0.05           # depth at which budget equivalents are interpolated
DEFAULT_OUTPUT = ROOT / "paper_tabs_figs/tables/tab_crossover.tex"
_REFIDX = re.compile(r"_ref(\d+)\.npy$")


def _ref_paths(t: int) -> list[Path]:
    """Reference-trajectory files for target t, ordered by reference index."""
    found: dict[int, Path] = {}
    for p in (EXP / f"model_{t}").glob("ref_losses_E*_*_ref*.npy"):
        m = _REFIDX.search(p.name)
        if m:
            found[int(m.group(1))] = p
    return [found[i] for i in sorted(found)]


def _gold(t: int):
    ar = EXP / f"model_{t}" / "attack_results"
    s = np.load(ar / "lira_refs_256_scores.npy").astype(np.float32)
    labels = np.load(ar / "lira_refs_256_labels.npy").astype(bool)
    thr = np.load(ar / "lira_refs_256_thresholds_by_fpr.npy",
                  allow_pickle=True).item()[FPR]
    return labels, (s >= float(thr)) & labels


def _precision(scores, members, vulnerable) -> list[float]:
    pr = tail_metrics_at_k(
        scores, members, K_FRACS, vulnerable=vulnerable,
    )
    return [float(pr.iloc[i]["precision"]) for i in range(len(K_FRACS))]


def collect() -> dict[str, np.ndarray]:
    """{row label: (T, len(K_FRACS)) precision}."""
    out: dict[str, list] = {f"$M={M}$": [] for M in M_GRID}
    out["LT-IQR ($M{=}0$)"] = []
    out["LT-BTM-R ($M{=}1$)"] = []
    out["\\rare{} ($M{=}1$)"] = []
    for t in TARGETS:
        members, vulnerable = _gold(t)
        refs = _ref_paths(t)
        if len(refs) < max(M_GRID):
            raise FileNotFoundError(
                f"target {t}: need {max(M_GRID)} reference trajectories, "
                f"found {len(refs)}")
        finals = [np.load(p).astype(np.float32)[:, -1] for p in refs[:max(M_GRID)]]
        for M in M_GRID:
            score = np.mean(finals[:M], axis=0).astype(np.float32)
            out[f"$M={M}$"].append(_precision(score, members, vulnerable))

        tgt = np.load(EXP / f"model_{t}" / "per_epoch_losses.npy").astype(np.float32).T
        out["LT-IQR ($M{=}0$)"].append(
            _precision(score_ltiqr_trajectory(tgt), members, vulnerable))
        ref0 = np.load(refs[0]).astype(np.float32)
        out["LT-BTM-R ($M{=}1$)"].append(
            _precision(score_avg_min_q_reference_losses([ref0], q=Q),
                       members, vulnerable))
        out["\\rare{} ($M{=}1$)"].append(
            _precision(load_rare_locked_scores(EXP, t, num_refs=1, q=Q),
                       members, vulnerable))
    return {k: np.asarray(v) for k, v in out.items()}


def budget_equivalent(curve_m: list[int], curve_p: list[float],
                      target: float) -> float | None:
    """Linear interpolation of M against the ranker's precision curve.

    Returns None if `target` lies outside the swept range.
    """
    if target <= curve_p[0]:
        return float(curve_m[0])
    for i in range(len(curve_m) - 1):
        lo, hi = curve_p[i], curve_p[i + 1]
        if lo <= target <= hi:
            frac = (target - lo) / (hi - lo) if hi > lo else 0.0
            return curve_m[i] + frac * (curve_m[i + 1] - curve_m[i])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    res = collect()
    mean = {k: v.mean(0) for k, v in res.items()}
    std = {k: v.std(0, ddof=1) for k, v in res.items()}

    upper = [f"$M={M}$" for M in M_GRID]
    lower = ["LT-IQR ($M{=}0$)", "LT-BTM-R ($M{=}1$)", "\\rare{} ($M{=}1$)"]
    ki = K_FRACS.index(EQUIV_K)

    def cell(lab, j):
        return pm_scriptstyle(mean[lab][j], std[lab][j], 3)

    lines = [
        "% Auto-generated by paper_tabs_figs/tab_crossover.py -- do not edit by hand.",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Budget equivalence on CIFAR10 under LiRA-256 gold. The upper",
        "block reports a standalone ranker that scores each member by its mean",
        "final-epoch loss over $M$ OUT reference models; the lower block reports",
        "the \\rare{} components and their combination, with their reference",
        "budgets. Mean$\\pm$std over 10 targets.}",
        "\\label{tab:crossover}",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        ADJUSTBOX_OPEN,
        "\\begin{tabular}{lccc}", "\\toprule",
        "\\textbf{Score} & \\textbf{P@1\\%} & \\textbf{P@3\\%} & \\textbf{P@5\\%} \\\\",
        "\\midrule",
    ]
    for lab in upper:
        lines.append(f"{lab:<6} & " + " & ".join(cell(lab, j) for j in range(3)) + " \\\\")
    lines.append("\\midrule")
    for lab in lower:
        lines.append(f"{lab:<21} & " + " & ".join(cell(lab, j) for j in range(3)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")

    with open(args.output.with_suffix(".csv"), "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["score", "k", "mean", "std"])
        for lab in upper + lower:
            for j, k in enumerate(K_LABELS):
                w.writerow([lab.replace("\\rare{}", "RARE"), k,
                            f"{mean[lab][j]:.4f}", f"{std[lab][j]:.4f}"])

    print("\n".join(lines))
    curve_p = [float(mean[f"$M={M}$"][ki]) for M in M_GRID]
    print("\n% ---- budget equivalents at P@5% (linear interpolation) ----")
    # Quoted only for the components that land in a well-conditioned part of
    # the curve. RARE sits past M=8, where the curve gains ~0.004/reference,
    # so its interpolated equivalent is not reportable -- see module docstring.
    for lab in ["LT-IQR ($M{=}0$)", "LT-BTM-R ($M{=}1$)"]:
        eq = budget_equivalent(M_GRID, curve_p, float(mean[lab][ki]))
        name = lab.split(" (")[0]
        got = f"M ~ {eq:.1f}" if eq is not None else f"beyond M={M_GRID[-1]}"
        print(f"%   {name:<12} P@5% = {mean[lab][ki]:.4f}  ->  {got}")
    rare_p = float(mean["\\rare{} ($M{=}1$)"][ki])
    print(f"%   RARE         P@5% = {rare_p:.4f}  ->  exceeds M={M_GRID[-1]} "
          f"({curve_p[-1]:.4f}); equivalent not quoted (near asymptote)")
    print(f"%   ranker curve P@5%: "
          + ", ".join(f"M={M}:{p:.4f}" for M, p in zip(M_GRID, curve_p)))

    # RARE vs the top row of the ranker, at every depth (quoted in the text).
    m_top = M_GRID[-1]
    print(f"\n% ---- \\rare{{}} (M=1) vs the M={m_top} ranker row ----")
    top = f"$M={m_top}$"
    rare_row = mean["\\rare{} ($M{=}1$)"]
    top_row = mean[top]
    for j, k in enumerate(K_LABELS):
        r, tp = float(rare_row[j]), float(top_row[j])
        print(f"%   {k:<6} RARE {r:.4f}  vs  M={m_top} {tp:.4f}   "
              f"delta {r - tp:+.4f} ({(r - tp) * 100:+.1f} pp)")


if __name__ == "__main__":
    main()
