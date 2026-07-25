"""Generate the CIFAR-10 Union-gold sweep LaTeX table (tab:union-cifar10).

Gold set: per target, the union of members detected by LiRA-256,
RMIA-256, and Attack-R-128 at the FPR threshold (default 1e-4), using the
thresholds saved by scripts/attacks.py. Reference attacks are swept over
M; LT-IQR / LOSS / RARE (M in {1,2,4}) are reported against the same
fixed gold.

Columns:
  AUC, TPR     attack-level metrics (TPR at the gold FPR), mean +/- std
               over the 10 targets; trajectory rankers (LT-IQR, RARE)
               have no member/non-member score and show ---.
  P@k          capped Precision @ k in {1,3,5}% against the Union gold.
  Speedup      cost(Union gold) / cost(method), from the same runtime-log
               cost model as tab:pipeline-cost. cost(Union gold) =
               cost(LiRA-256) + cost(RMIA-256) + cost(Attack-R-128).
               RARE's per-reference cost uses the logged retrain time of
               the specific reference index it loads (ref0 = full
               sharedinit ~145 s, ref1..3 = boot0.50 ~92 s).

Usage:
  python paper_tabs_figs/tab_union.py
  python paper_tabs_figs/tab_union.py --fpr 1e-3
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc as sk_auc, roc_curve

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts"), str(Path(__file__).parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import tab_pipeline_cost as tpc  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "main_table_mod", ROOT / "scripts" / "main_table.py")
_mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mt)

DEFAULT_EXP = tpc.DEFAULT_BENCHMARKS["CIFAR10"]
DEFAULT_TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
DEFAULT_OUTPUT = ROOT / "paper_tabs_figs/figures/tab_union_cifar10.tex"
K_FRACS = [0.01, 0.03, 0.05]
UNION_ATTACKS = [("lira_refs", 256), ("rmia_refs", 256), ("attackr_refs", 128)]
SWEEP = {"LiRA": [2, 4, 8, 16, 32, 64, 128, 256],
         "RMIA": [2, 4, 8, 16, 32, 64, 128, 256],
         "Attack-R": [2, 4, 8, 16, 32, 64, 128]}
GEN_AUG = {"LiRA": 2, "RMIA": 1, "Attack-R": 1}
RARE_MS = [1, 2, 4]
_REFIDX_RE = re.compile(r"target_(\d+)_ref_(\d+)$")


def union_gold(exp: Path, t: int, fpr: float):
    ar = exp / f"model_{t}" / "attack_results"
    labels = np.load(ar / "lira_refs_256_labels.npy").astype(bool)
    hit = np.zeros_like(labels)
    for prefix, m in UNION_ATTACKS:
        scores = np.load(ar / f"{prefix}_{m}_scores.npy").astype(np.float32)
        thr = np.load(ar / f"{prefix}_{m}_thresholds_by_fpr.npy",
                      allow_pickle=True).item()[fpr]
        hit |= scores >= float(thr)
    return labels, hit & labels


def attack_auc_tpr(scores, labels, fpr_target):
    fpr, tpr, _ = roc_curve(labels.astype(int), scores)
    idx = np.where(fpr <= fpr_target)[0]
    return sk_auc(fpr, tpr), (tpr[idx[-1]] if idx.size else 0.0)


def ref_costs(exp: Path):
    """Mean logged retrain seconds per RARE reference index."""
    rows = tpc._load_runtime_rows(exp / "runtime.csv")
    by_idx = {}
    for r in rows:
        if r["phase"] != "ref_retrain":
            continue
        m = _REFIDX_RE.search(r["item"])
        if m:
            by_idx.setdefault(int(m.group(2)), []).append(r)
    return {i: float(np.mean([r["sec"] for r in tpc._dedup_min(v)]))
            for i, v in by_idx.items()}


def build_costs(exp: Path):
    c = tpc.collect(exp)
    refs = ref_costs(exp)

    def score_time(method, m):
        logged = {mm: s.mean for (meth, mm), s in c["score"].items()
                  if meth == method}
        if not logged:
            return 0.0
        nearest = min(logged, key=lambda mm: abs(mm - m))
        return logged[nearest]

    def attack_cost(method, m):
        aug = GEN_AUG[method]
        return (m * c["base"].mean + (m + 1) * c["gen"][aug].mean
                + score_time(method, m))

    def rare_cost(m):
        return (c["rare_oh"].mean + sum(refs[r] for r in range(m))
                + c["rare_score"].mean)

    union = (attack_cost("LiRA", 256) + attack_cost("RMIA", 256)
             + attack_cost("Attack-R", 128))
    return c, attack_cost, rare_cost, union


def _fmt(mean, std, dec):
    return f"{mean:.{dec}f}{{\\scriptsize$\\pm${std:.{dec}f}}}"


def _fmt_speedup(v):
    body = f"{v:,.1f}".replace(",", "{,}") if v < 10000 else \
        f"{v:,.0f}".replace(",", "{,}")
    return f"${body}\\times$"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--fpr", type=float, default=1e-4)
    ap.add_argument("--targets", type=int, nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    exp = args.experiment_dir

    gold = {t: union_gold(exp, t, args.fpr) for t in args.targets}
    c, attack_cost, rare_cost, union_cost = build_costs(exp)

    def evaluate(loader, with_attack_metrics):
        """Per-target AUC/TPR and capped P@k against the Union gold."""
        aucs, tprs, pks = [], [], []
        for t in args.targets:
            labels, V = gold[t]
            scores = loader(t)
            if with_attack_metrics:
                a, tp = attack_auc_tpr(scores, labels, args.fpr)
                aucs.append(a); tprs.append(tp)
            pr = _mt.tail_metrics_at_k(scores, labels, K_FRACS, vulnerable=V)
            pks.append([float(pr.iloc[i]["precision"])
                        for i in range(len(K_FRACS))])
        pks = np.asarray(pks)
        return (np.asarray(aucs), np.asarray(tprs)) if with_attack_metrics \
            else (None, None), pks

    rows = []  # (label, auc_stat|None, tpr_stat|None, pk mean, pk std, speedup)
    for method, ms in SWEEP.items():
        for m in ms:
            (aucs, tprs), pks = evaluate(
                lambda t, M=m: _mt._load_attack_scores(exp, t, method, M), True)
            rows.append((f"{method} (M={m})", aucs, tprs,
                         pks.mean(0), pks.std(0, ddof=1),
                         union_cost / attack_cost(method, m)))
        rows.append(None)  # \midrule
    (aucs, tprs), pks = evaluate(lambda t: _mt._load_liu_mlp(exp, t, 1), True)
    lt_mlp_cost = (c["base"].mean + 2 * c["loss_oh"].mean
                   + c["liu_score"].mean)
    rows.append(("LT-MLP (M=1)", aucs, tprs, pks.mean(0), pks.std(0, ddof=1),
                 union_cost / lt_mlp_cost))
    rows.append(None)
    _, pks = evaluate(lambda t: _mt._load_lt_iqr(exp, t), False)
    rows.append(("LT-IQR (M=0)", None, None, pks.mean(0), pks.std(0, ddof=1),
                 union_cost / (c["loss_oh"].mean + c["lt_iqr_score"].mean)))
    rows.append(None)
    (aucs, tprs), pks = evaluate(lambda t: _mt._load_loss_scores(exp, t), True)
    loss_cost = c["loss_fwd"].mean + c["score"][("LOSS", 0)].mean
    rows.append(("LOSS (M=0)", aucs, tprs, pks.mean(0), pks.std(0, ddof=1),
                 union_cost / loss_cost))
    rows.append(None)
    for m in RARE_MS:
        _, pks = evaluate(lambda t, M=m: _mt._load_rare(exp, t, M), False)
        rows.append((f"\\rare{{}} (M={m})", None, None,
                     pks.mean(0), pks.std(0, ddof=1),
                     union_cost / rare_cost(m)))

    fpr_tex = f"10^{{{int(np.log10(args.fpr))}}}"
    lines = [
        "% Auto-generated by paper_tabs_figs/tab_union.py -- do not edit by hand.",
        "\\begin{table}[t!]",
        "\\centering",
        "\\caption{CIFAR10 sweep under the fixed Union gold set, defined as the"
        " union of vulnerable members detected by LiRA-256, RMIA-256, and"
        f" Attack-R-128 at FPR$\\leq {fpr_tex}$. \\rare{{}} is shown for"
        " $M\\in\\{1,2,4\\}$ with $q=50\\%$, and reference attacks are swept"
        " over $M$. Speedup is relative to constructing the full Union gold"
        " set.}",
        "\\label{tab:union-cifar10}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l c c c c c c}",
        "\\toprule",
        " & \\multicolumn{2}{c}{\\textbf{MIA performance}} &"
        " \\multicolumn{3}{c}{\\textbf{Precision@k}} &"
        " \\multirow{2}{*}{\\textbf{Speedup}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-6}",
        " \\textbf{Method} & \\textbf{AUC} & \\textbf{TPR}"
        " & k=1\\% & k=3\\% & k=5\\% &  \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row is None:
            lines.append("\\midrule")
            continue
        label, aucs, tprs, pk_m, pk_s, speedup = row
        auc_cell = (f"{aucs.mean():.2f}"
                    f"{{\\scriptsize$\\pm${aucs.std(ddof=1):.3f}}}"
                    if aucs is not None else "---")
        tpr_cell = _fmt(tprs.mean(), tprs.std(ddof=1), 3) if tprs is not None else "---"
        cells = [label, auc_cell, tpr_cell]
        cells += [_fmt(pk_m[i], pk_s[i], 2) for i in range(len(K_FRACS))]
        cells.append(_fmt_speedup(speedup))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]

    tex = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex + "\n")
    print(tex)

    print("\n% ---- prose numbers ----")
    print(f"% Union gold cost: {union_cost:,.0f} s; mean |V_union| = "
          f"{np.mean([gold[t][1].sum() for t in args.targets]):.0f}")
    for row in rows:
        if row and row[0].startswith("\\rare"):
            print(f"% {row[0]}: P@1/3/5% = "
                  + " / ".join(f"{v:.2f}" for v in row[3])
                  + f"; speedup {row[5]:,.1f}x; cost ratio vs M=1: "
                  f"{rare_cost(int(row[0][-2])) / rare_cost(1):.2f}x")


if __name__ == "__main__":
    main()
