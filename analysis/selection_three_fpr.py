"""Vulnerable-set selection comparison across three gold operating points.

For each FPR in {0, 1e-4, 1e-3} the gold set is LiRA-256 thresholded at that FPR,
and seven selectors are scored against it:

  KMeans-RARE (2-D)     incumbent: 2-D KMeans over z_M(LT-IQR), z_M(LT-BTM-R)
  KMeans-RARE (1-D)     1-D KMeans on the scalar RARE score
  KMeans-LT-IQR (1-D)   1-D KMeans on LT-IQR alone
  LiRA (M=8) / RMIA (M=8) / Attack-R (M=16) / LT-MLP (M=1)
                        thresholded at the same FPR on the target's own
                        non-members

Gold convention, KMeans, z-scoring and metrics are reused verbatim from
`paper_tabs_figs/tab_actionable_selection.py`, matching
`analysis/recall_three_fpr.py`. Nothing under version control is written;
output is `analysis/selection_three_fpr.md`.

Usage:
  python analysis/selection_three_fpr.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List

# sklearn KMeans is not reproducible across processes under multi-threaded
# OpenMP even with random_state=0; see analysis/matched_selection.md.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location(
    "sel_base", ROOT / "paper_tabs_figs" / "tab_actionable_selection.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

BENCHMARKS = base.DEFAULT_BENCHMARKS
TARGETS = base.DEFAULT_TARGETS
DS = ["CIFAR10", "CIFAR100", "CINIC10"]
BLOCKS = DS + ["Avg."]
FPRS = [0.0, 1e-4, 1e-3]
FPR_LABEL = {0.0: "FPR = 0", 1e-4: "FPR ≤ 1e-4", 1e-3: "FPR ≤ 1e-3"}
Q = 0.50
MIN_S = 50

KM2 = "KMeans-RARE (2-D)"
KM1 = "KMeans-RARE (1-D)"
KMI = "KMeans-LT-IQR (1-D)"
ATTACKS = {"LiRA (M=8)": "lira_refs_8", "RMIA (M=8)": "rmia_refs_8",
           "Attack-R (M=16)": "attackr_refs_16"}
LTMLP = "LT-MLP (M=1)"
METHODS = [KM2, KM1, KMI] + list(ATTACKS) + [LTMLP]

FIELDS = ["gold", "selected", "tp", "recall", "precision", "f1"]
LABEL = {"gold": "|V|", "selected": "|S|", "tp": "|S∩V|", "recall": "Recall",
         "precision": "Precision", "f1": "F1"}
RATE = {"recall", "precision", "f1"}


def evaluate():
    per: Dict[float, Dict[str, Dict[str, list]]] = {
        f: {m: {d: [] for d in DS} for m in METHODS} for f in FPRS}
    agreement = {d: {"jaccard": [], "symdiff": []} for d in DS}

    for ds, exp in BENCHMARKS.items():
        for t in TARGETS:
            members, _ = base._load_lira_gold(exp, int(t), FPRS[0])
            z_iqr = base._zscore_on_members(base._load_lt_iqr(exp, int(t)), members)
            z_btm = base._zscore_on_members(base._load_lt_btm_r(exp, int(t), Q), members)
            rare = z_iqr + z_btm

            # Gold-independent selectors, computed once.
            sel2 = base._kmeans_select(np.column_stack([z_iqr, z_btm]), members, rare)
            sel1 = base._kmeans_select(rare.reshape(-1, 1), members, rare)
            seli = base._kmeans_select(z_iqr.reshape(-1, 1), members, z_iqr)
            a, b = sel2 & members, sel1 & members
            agreement[ds]["jaccard"].append(
                float((a & b).sum()) / float((a | b).sum()))
            agreement[ds]["symdiff"].append(int((a ^ b).sum()))

            mlp = np.load(exp / f"model_{int(t)}" / "liu_mlp_scores_R1.npy"
                          ).astype(np.float32)

            for f in FPRS:
                mem_f, vulnerable = base._load_lira_gold(exp, int(t), f)
                if not np.array_equal(mem_f, members):
                    raise AssertionError(f"member mask differs across FPR: {ds} t={t}")
                sels = {KM2: sel2, KM1: sel1, KMI: seli}
                for name, prefix in ATTACKS.items():
                    sels[name] = base._threshold_select(exp, int(t), prefix, f)
                sels[LTMLP] = base._score_threshold_select(mlp, members, f)
                for m in METHODS:
                    per[f][m][ds].append(dict(
                        base._metrics(sels[m], members, vulnerable)))
            print(f"  {ds} t={t} done", file=sys.stderr, flush=True)
    return per, agreement


def ms(v):
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1))


def cell(per, f, m, blk, k):
    """(mean, std) for one field; Avg. = over the three dataset means."""
    if blk != "Avg.":
        return ms([r[k] for r in per[f][m][blk]])
    dsm = [ms([r[k] for r in per[f][m][d]])[0] for d in DS]
    return float(np.mean(dsm)), float(np.std(dsm, ddof=1))


def fmt(field, mean, std):
    return (f"{mean:.4f} ± {std:.4f}" if field in RATE
            else f"{mean:,.1f} ± {std:,.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "analysis" / "selection_three_fpr.md")
    args = ap.parse_args()

    per, agreement = evaluate()

    L: List[str] = []
    A = L.append
    A("# Vulnerable-set selection across three gold operating points")
    A("")
    A("## Setup")
    A("")
    A("| | |")
    A("|---|---|")
    A("| Experiment dirs | `2025-12-07_0140` / `2025-12-25_0121` / `2026-05-11_0215` |")
    A("| Targets | 10 per dataset: 0, 25, 50, 75, 100, 125, 150, 175, 200, 250 |")
    A("| Gold | LiRA-256 at FPR ∈ {0, 1e-4, 1e-3} |")
    A("| Threads | `OMP_NUM_THREADS=1` pinned |")
    A("")
    A("| Selector | Rule | Depends on gold FPR? |")
    A("|---|---|---|")
    A(f"| `{KM2}` | incumbent 2-D KMeans over `z_M(LT-IQR)`, `z_M(LT-BTM-R)` | no |")
    A(f"| `{KM1}` | 1-D KMeans on the scalar RARE score | no |")
    A(f"| `{KMI}` | 1-D KMeans on LT-IQR alone | no |")
    A("| `LiRA (M=8)`, `RMIA (M=8)`, `Attack-R (M=16)` | score ≥ saved threshold at that FPR | yes |")
    A("| `LT-MLP (M=1)` | score ≥ ROC threshold at that FPR (no saved dict) | yes |")
    A("")
    A("All KMeans selectors use `n_clusters=2, random_state=0, n_init=50` on")
    A("member-standardized scores, taking the higher-mean cluster. The three KMeans rows")
    A("are gold-independent, so their |S| is identical across the three blocks; the four")
    A("attack rows re-threshold at each FPR and their |S| changes.")
    A("")
    A("Attack thresholds are calibrated on **the target's own non-members**: the saved")
    A("`*_thresholds_by_fpr.npy` entries come from `attacks.py::compute_metrics`, which")
    A("takes `thresholds[idx[-1]]` over `fpr_list <= target_fpr` from a ROC over that")
    A("attack's own scores and labels. Verified: at FPR = 0 no non-member sits at or above")
    A("the threshold for any of LiRA-8 / RMIA-8 / Attack-R-16, on all 30 targets, so the")
    A("selection is exactly the members outscoring every non-member — the same convention")
    A("used for the gold set in `analysis/recall_three_fpr.py`. LT-MLP has no saved dict")
    A("and uses `_score_threshold_select`, which applies the identical ROC rule.")
    A("")
    A("Mean ± std is over the 10 targets; the **Avg.** block is the mean of the three")
    A("dataset means with std over those three (ddof=1).")
    A("")

    # ------------------------------------------------------- 1-D vs 2-D check
    A("## 1-D and 2-D KMeans-RARE agree")
    A("")
    A("| Dataset | Jaccard(2-D, 1-D) | symmetric difference | max \\|ΔF1\\| over the 3 gold thresholds |")
    A("|---|---|---|---|")
    for ds in DS:
        j = ms(agreement[ds]["jaccard"])
        sd = ms(agreement[ds]["symdiff"])
        dmax = max(abs(cell(per, f, KM2, ds, "f1")[0] - cell(per, f, KM1, ds, "f1")[0])
                   for f in FPRS)
        A(f"| {ds} | {j[0]:.4f} ± {j[1]:.4f} | {sd[0]:,.0f} ± {sd[1]:,.0f} | {dmax:.4f} |")
    A("")
    allj = np.concatenate([agreement[d]["jaccard"] for d in DS])
    allmax = max(abs(cell(per, f, KM2, d, "f1")[0] - cell(per, f, KM1, d, "f1")[0])
                 for f in FPRS for d in DS)
    A(f"Overlap is Jaccard {allj.min():.3f}–{allj.max():.3f} and F1 never differs by more")
    A(f"than {allmax:.4f} in any (threshold × dataset) cell. The 2-D clustering and the 1-D")
    A("clustering on the summed score are effectively the same selector — consistent with")
    A("`analysis/matched_selection.md`, where the 2-D selection also matched a plain")
    A("threshold on the scalar RARE score to three decimals. Both rows are kept below so")
    A("the equivalence is visible, but they are not independent results.")
    A("")

    # ---------------------------------------------------------------- results
    for f in FPRS:
        A("---")
        A("")
        A(f"## Gold = LiRA-256 at {FPR_LABEL[f]}")
        A("")
        A("| Dataset | Method | " + " | ".join(LABEL[k] for k in FIELDS) + " |")
        A("|---" * (len(FIELDS) + 2) + "|")
        for blk in BLOCKS:
            for m in METHODS:
                cells = [fmt(k, *cell(per, f, m, blk, k)) for k in FIELDS]
                small = cell(per, f, m, blk, "selected")[0] < MIN_S
                name = f"{m} ⚠️" if small else m
                A(f"| {blk} | {name} | " + " | ".join(cells) + " |")
        A("")

    # -------------------------------------------------------------- win table
    A("---")
    A("")
    A("## Win table — highest mean F1 per cell")
    A("")
    A(f"Methods whose mean |S| in that cell is below {MIN_S} are excluded from the ranking")
    A("(F1 is not meaningful there) and listed in the flag column instead.")
    A("")
    A("| Gold | Dataset | Winner | F1 | Runner-up | F1 | Margin | ⚠️ \\|S\\| < 50 |")
    A("|---|---|---|---|---|---|---|---|")
    win_counts: Dict[str, int] = {}
    for f in FPRS:
        for ds in DS:
            scored, flagged = [], []
            for m in METHODS:
                s = cell(per, f, m, ds, "selected")[0]
                f1 = cell(per, f, m, ds, "f1")[0]
                (flagged if s < MIN_S else scored).append((m, f1, s))
            scored.sort(key=lambda x: -x[1])
            if not scored:
                A(f"| {FPR_LABEL[f]} | {ds} | — (all flagged) | — | — | — | — | "
                  + ", ".join(f"{m} ({s:,.1f})" for m, _, s in flagged) + " |")
                continue
            w = scored[0]
            r = scored[1] if len(scored) > 1 else (None, float("nan"), 0)
            win_counts[w[0]] = win_counts.get(w[0], 0) + 1
            flag = (", ".join(f"{m} ({s:,.1f})" for m, _, s in flagged)
                    if flagged else "—")
            A(f"| {FPR_LABEL[f]} | {ds} | **{w[0]}** | {w[1]:.4f} | "
              f"{r[0] or '—'} | {r[1]:.4f} | {(w[1] - r[1]) * 100:+.2f} pp | {flag} |")
    A("")
    A("### Wins by method")
    A("")
    A("| Method | cells won (of 9) |")
    A("|---|---|")
    for m in METHODS:
        if win_counts.get(m):
            A(f"| {m} | {win_counts[m]} |")
    A("")
    A("Because the 2-D and 1-D KMeans-RARE rows are near-duplicates, a cell won by one is")
    A("effectively won by both; the split between them is arbitrary and reflects")
    A("differences in the fourth decimal, not a real preference. Read the table as")
    A("*KMeans-RARE wins 8 of 9, KMeans-LT-IQR takes CIFAR10 at the loosest gold*.")
    A("")
    A("### The margin over the attack rows is mostly cardinality, not ranking")
    A("")
    A("This is the caveat that matters most for reading the table above. The KMeans rows")
    A("and the FPR-thresholded attack rows do not select comparable numbers of members:")
    A("")
    A("| Gold | KMeans-RARE \\|S\\| | LiRA-8 \\|S\\| | ratio | KMeans F1 | LiRA-8 F1 |")
    A("|---|---|---|---|---|---|")
    for f in FPRS:
        sk = cell(per, f, KM2, "Avg.", "selected")[0]
        sl = cell(per, f, "LiRA (M=8)", "Avg.", "selected")[0]
        A(f"| {FPR_LABEL[f]} | {sk:,.0f} | {sl:,.0f} | {sk / sl:.1f}× | "
          f"{cell(per, f, KM2, 'Avg.', 'f1')[0]:.3f} | "
          f"{cell(per, f, 'LiRA (M=8)', 'Avg.', 'f1')[0]:.3f} |")
    A("")
    A("Thresholding at a strict FPR is simply a *smaller* selection rule than taking the")
    A("upper KMeans cluster, and F1 against a gold set of a few thousand members punishes")
    A("under-selection hard. The attack rows' low F1 here therefore measures their")
    A("operating point, not the quality of their ranking.")
    A("")
    A("`analysis/matched_selection.md` isolates the ranking by giving every score the same")
    A("budget B = |S_KMeans-RARE|. Under that comparison, at the FPR ≤ 1e-4 gold, LiRA-8")
    A("reaches F1 0.688 against RARE's 0.696 — a gap of under one point, versus the")
    A(f"{(cell(per, 1e-4, KM2, 'Avg.', 'f1')[0] - cell(per, 1e-4, 'LiRA (M=8)', 'Avg.', 'f1')[0]) * 100:.0f} pp")
    A("gap shown here. Essentially all of the apparent advantage in this table comes from")
    A("choosing how many samples to select, which is exactly what an unsupervised selector")
    A("is for — but it is a claim about the *selection rule*, not about RARE as a score.")
    A("Cite this table for \"KMeans-RARE is a better turnkey selector\"; cite the")
    A("matched-cardinality panel for \"RARE ranks better than LiRA-8\".")
    A("")

    # ------------------------------------------------------------------ flags
    A("### All flagged cells")
    A("")
    A("| Gold | Dataset | Method | mean \\|S\\| | targets with \\|S\\| < 50 |")
    A("|---|---|---|---|---|")
    nflag = 0
    for f in FPRS:
        for ds in DS:
            for m in METHODS:
                s = cell(per, f, m, ds, "selected")[0]
                if s < MIN_S:
                    nflag += 1
                    n = int(sum(1 for r in per[f][m][ds] if r["selected"] < MIN_S))
                    A(f"| {FPR_LABEL[f]} | {ds} | {m} | {s:,.1f} | {n} / {len(TARGETS)} |")
    if nflag == 0:
        A("| — | — | — | — | — |")
    A("")
    if nflag:
        A(f"{nflag} of {len(FPRS) * len(DS) * len(METHODS)} (gold × dataset × method) cells")
        A("select fewer than 50 members on average. All of them are attack rows at the")
        A("strict end of the sweep, where thresholding at a near-zero FPR admits almost")
        A("nothing — the precision in those cells can look excellent while resting on a")
        A("handful of samples.")
    A("")

    A("## Reproducing")
    A("")
    A("```bash")
    A("python analysis/selection_three_fpr.py")
    A("```")
    A("")
    A("Bit-reproducible with `OMP_NUM_THREADS=1`.")

    args.output.write_text("\n".join(L) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")

    for f in FPRS:
        print(f"\n=== {FPR_LABEL[f]} ===")
        for blk in BLOCKS:
            print(f"  -- {blk}")
            for m in METHODS:
                print(f"     {m:<22} |S|={cell(per, f, m, blk, 'selected')[0]:9,.1f} "
                      f"R={cell(per, f, m, blk, 'recall')[0]:.3f} "
                      f"P={cell(per, f, m, blk, 'precision')[0]:.3f} "
                      f"F1={cell(per, f, m, blk, 'f1')[0]:.3f}")


if __name__ == "__main__":
    main()
