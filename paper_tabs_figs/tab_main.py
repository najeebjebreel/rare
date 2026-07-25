"""Generate the cross-dataset main results LaTeX table (tab:main).

Rebuilds every cell from on-disk artifacts:

  P@k cells   capped Precision @ k in {1, 3, 5}% under LiRA-256 gold
              (hits at the FPR threshold, intersected with members),
              mean +/- std over the 10 targets per dataset; the Avg
              column is the mean +/- std over the three dataset means.
  Speedup     cross-dataset mean of the per-dataset pipeline speedups
              from paper_tabs_figs/tab_pipeline_cost.py (same runtime
              logs as tab:pipeline-cost) -- relative to LiRA-256.

Best per column is bold, second best underlined.

Usage:
  python paper_tabs_figs/tab_main.py
  python paper_tabs_figs/tab_main.py --fpr 1e-3 --output figures/tab.tex
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts"), str(Path(__file__).parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import tab_pipeline_cost as tpc  # noqa: E402  (sibling module)

_spec = importlib.util.spec_from_file_location(
    "main_table_mod", ROOT / "scripts" / "main_table.py")
_mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mt)

DEFAULT_TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
DEFAULT_OUTPUT = ROOT / "paper_tabs_figs/figures/tab_main.tex"
K_FRACS = [0.01, 0.03, 0.05]
K_LABELS = ["P@1\\%", "P@3\\%", "P@5\\%"]

# (paper label, pipeline-cost label for the speedup column, score loader)
METHODS = [
    ("LiRA (M=8)", "LiRA (M=8)",
     lambda exp, t: _mt._load_attack_scores(exp, t, "LiRA", 8)),
    ("RMIA (M=8)", "RMIA (M=8)",
     lambda exp, t: _mt._load_attack_scores(exp, t, "RMIA", 8)),
    ("Attack-R (M=16)", "Attack-R (M=16)",
     lambda exp, t: _mt._load_attack_scores(exp, t, "Attack-R", 16)),
    ("LT-MLP (M=1)", "LT-MLP (M=1)",
     lambda exp, t: _mt._load_liu_mlp(exp, t, 1)),
    ("LT-IQR (M=0)", "LT-IQR (M=0)",
     lambda exp, t: _mt._load_lt_iqr(exp, t)),
    ("LOSS (M=0)", "LOSS (M=0)",
     lambda exp, t: _mt._load_loss_scores(exp, t)),
    ("\\rare{} (M=1)", "RARE (M=1)",
     lambda exp, t: _mt._load_rare(exp, t, 1)),
]
GROUP_BREAKS = {"LT-IQR (M=0)", "\\rare{} (M=1)"}  # \midrule before these rows


def precision_at_k(exp: Path, targets, method_loader, fpr: float):
    """Per-target capped P@k under LiRA-256 gold; returns (T, len(K_FRACS))."""
    out = []
    for t in targets:
        gold_scores, members, vulnerable = _mt._load_lira_gold(exp, t, fpr)
        scores = method_loader(exp, t)
        pr = _mt.tail_metrics_at_k(
            scores, members, K_FRACS,
            vulnerable=vulnerable, gold_scores=gold_scores,
        )
        out.append([float(pr.iloc[i]["precision"]) for i in range(len(K_FRACS))])
    return np.asarray(out)


def speedups():
    """{pipeline label: cross-dataset mean speedup vs LiRA-256}."""
    per_ds = {name: tpc.build_rows(tpc.collect(exp))
              for name, exp in tpc.DEFAULT_BENCHMARKS.items()}
    gold = {n: rows["LiRA (M=256, gold)"][3].mean for n, rows in per_ds.items()}
    out = {}
    for label in set(m[1] for m in METHODS):
        out[label] = float(np.mean(
            [gold[n] / per_ds[n][label][3].mean for n in per_ds]))
    return out


def _rank_decorations(col_means):
    """{row index: 'textbf'|'underline'} for the best / second best."""
    order = np.argsort(-np.asarray(col_means))
    deco = {int(order[0]): "textbf"}
    if len(order) > 1:
        deco[int(order[1])] = "underline"
    return deco


def _fmt_p(mean: float, std: float, deco: str | None) -> str:
    body = f"{mean:.2f}"
    if deco:
        body = f"\\{deco}{{{body}}}"
    return f"{body}{{\\scriptsize$\\pm${std:.2f}}}"


def _fmt_speedup(v: float, deco: str | None) -> str:
    body = f"{v:,.0f}".replace(",", "{,}") if v >= 1000 else f"{v:.1f}"
    if deco:
        body = f"\\{deco}{{{body}}}"
    return f"{body}$\\times$"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fpr", type=float, default=1e-4,
                    help="FPR defining the LiRA-256 gold set (default 1e-4)")
    ap.add_argument("--targets", type=int, nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    ds_names = list(tpc.DEFAULT_BENCHMARKS)
    # stats[method][dataset] = (means over targets, stds) per k; Avg appended.
    cell_mean = np.zeros((len(METHODS), (len(ds_names) + 1) * len(K_FRACS)))
    cell_std = np.zeros_like(cell_mean)
    for mi, (_, _, loader) in enumerate(METHODS):
        ds_means = []
        for di, name in enumerate(ds_names):
            exp = tpc.DEFAULT_BENCHMARKS[name]
            pk = precision_at_k(exp, args.targets, loader, args.fpr)
            cell_mean[mi, di * 3:di * 3 + 3] = pk.mean(axis=0)
            cell_std[mi, di * 3:di * 3 + 3] = pk.std(axis=0, ddof=1)
            ds_means.append(pk.mean(axis=0))
        ds_means = np.asarray(ds_means)               # (3 datasets, 3 ks)
        cell_mean[mi, -3:] = ds_means.mean(axis=0)
        cell_std[mi, -3:] = ds_means.std(axis=0, ddof=1)

    speed = speedups()
    speed_col = [speed[m[1]] for m in METHODS]

    col_deco = [_rank_decorations(cell_mean[:, c]) for c in range(cell_mean.shape[1])]
    speed_deco = _rank_decorations(speed_col)

    header = [
        "% Auto-generated by paper_tabs_figs/tab_main.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Main results on CIFAR10, CIFAR100, and CINIC10 under LiRA-256"
        " gold. \\rare{} uses $M{=}1$ and $q=50\\%$; reference baselines use the"
        " budgets shown in the method names.}",
        "\\label{tab:main}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l ccc ccc ccc ccc r}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Method}}",
    ]
    for name in ds_names + ["Avg."]:
        header.append(f"  & \\multicolumn{{3}}{{c}}{{\\textbf{{{name}}}}}")
    header += [
        "  & \\multirow{2}{*}{\\textbf{Speedup}} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}",
        " " + " ".join(f"& \\textbf{{{k}}}" for _ in range(4) for k in K_LABELS) + " & \\\\",
        "\\midrule",
    ]
    lines = header
    for mi, (label, _, _) in enumerate(METHODS):
        if label in GROUP_BREAKS:
            lines.append("\\midrule")
        cells = [label]
        for c in range(cell_mean.shape[1]):
            cells.append(_fmt_p(cell_mean[mi, c], cell_std[mi, c],
                                col_deco[c].get(mi)))
        cells.append(_fmt_speedup(speed_col[mi], speed_deco.get(mi)))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]

    tex = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex + "\n")
    print(tex)

    # Numbers referenced in the running text of sec:results-lira-gold.
    ri = next(i for i, m in enumerate(METHODS) if m[1] == "RARE (M=1)")
    li = next(i for i, m in enumerate(METHODS) if m[0] == "LiRA (M=8)")
    print("\n% ---- prose numbers ----")
    for j, k in enumerate(["P@1%", "P@3%", "P@5%"]):
        print(f"% Avg {k}: RARE {cell_mean[ri, -3 + j]:.2f}"
              f"+/-{cell_std[ri, -3 + j]:.2f}; "
              f"LiRA-8 {cell_mean[li, -3 + j]:.2f}; "
              f"delta {(cell_mean[ri, -3 + j] - cell_mean[li, -3 + j]) * 100:+.1f} pp")
    print(f"% Speedup: RARE {speed_col[ri]:.1f}x vs LiRA-8 {speed_col[li]:.1f}x "
          f"(ratio {speed_col[ri] / speed_col[li]:.1f}x)")


if __name__ == "__main__":
    main()
