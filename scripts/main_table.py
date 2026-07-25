"""Per-benchmark main-table generator for the RARE paper.

For one experiment directory, loads on-disk scores for every method, computes
per-target metrics, aggregates mean ± CI95 over the 10 targets, and emits
CSV + booktabs LaTeX.

Row structure (fixed order; M = # additional models, shadow or reference):
  1.  LiRA       (M=256, gold)
  2.  LiRA       (M=X1_closeness)
  3.  LiRA       (M=X1_cost_match)
  4.  RMIA       (M=X3_closeness)
  5.  RMIA       (M=X3_cost_match)
  6.  Attack-R   (M=X2_closeness)
  7.  Attack-R   (M=X2_cost_match)
  8.  Liu-MLP    (M=Xliu_closeness)
  9.  Liu-MLP    (M=Xliu_cost_match)
  10. LT-IQR     (single row, target-only)
  11. LOSS       (single row, target-only)
  12. RARE      (M=1)
  13. RARE      (M=2)
  14. RARE      (M=4)   -- only if four reference trajectories exist

Metrics in THIS table (the trade-off table) — mean ± CI95 over 10 targets:
  - P @ k = {1%, 3%, 5%, 10%} at FPR = 1e-4
  - Tail-ρ vs LiRA-256
  - Cost (s) — per-target audit cost from the no-amortization model

AUC, TPR@FPR live in a separate "attack-performance" table (not generated here).

The closeness / cost-match X values are auto-read from
<exp>/budget_match/budget_match.csv.

Outputs (under <exp>/main_table/):
  per_target.csv         -- (target, method, metric, value)
  main_table.csv         -- long: (method, metric, mean, ci95)
  main_table_wide.csv    -- one row per method, columns = metrics
  main_table.tex         -- booktabs paper-ready
"""
import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rare.analysis.utils import load_rare_locked_scores, tail_metrics_at_k


FPR_DEFAULT = 1e-4   # gold-V FPR default for the capped Precision and Tail-ρ
K_FRACS = [0.01, 0.03, 0.05, 0.10]
DEFAULT_TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
LIU_MLP_R_DEFAULT = 4  # only used as a fallback if budget_match has no Liu-MLP row

# Cost model — must match budget_match_sweep.py
def _rare_cost(M: int) -> float:
    return 24.0 + M * 144.0

RARE_M1_COST = _rare_cost(1)   # 168
RARE_M2_COST = _rare_cost(2)   # 312
RARE_M4_COST = _rare_cost(4)   # 600
LOSS_COST    = 1.5
LT_IQR_COST  = 24.0   # target trace overhead + np.quantile (essentially free)
LIRA_M256_COST = 256 * 121.5 + 2.7  # = 31,106 s


def _cost(method: str, M: int) -> float:
    if method == "LiRA":     return M * 121.5 + 2.7
    if method == "Attack-R": return M * 121.5 + 5.5
    if method == "RMIA":     return M * 121.5 + 11.5
    if method == "Liu-MLP":  return M * 144.0 + 5.0
    raise ValueError(method)


METRIC_ORDER = [
    "P@1%", "P@3%", "P@5%", "P@10%",
    "Cost(s)",
]
METRIC_LATEX = {
    "P@1%":     r"P@1\%",
    "P@3%":     r"P@3\%",
    "P@5%":     r"P@5\%",
    "P@10%":    r"P@10\%",
    "Cost(s)":  r"Cost (s)",
}


def _load_lira_gold(exp: Path, t: int, fpr: float):
    ar = exp / f"model_{t}" / "attack_results"
    scores = np.load(ar / "lira_refs_256_scores.npy").astype(np.float32)
    labels = np.load(ar / "lira_refs_256_labels.npy").astype(bool)
    thr_dict = np.load(
        ar / "lira_refs_256_thresholds_by_fpr.npy", allow_pickle=True
    ).item()
    if fpr not in thr_dict:
        raise KeyError(f"FPR={fpr} not in LiRA thresholds (target {t})")
    thr = float(thr_dict[fpr])
    vulnerable = (scores >= thr) & labels
    return scores, labels, vulnerable


def _per_target_metrics(scores, members, vulnerable, lira_gold_scores):
    """Capped Precision @ {1,3,5,10}% and Tail-ρ vs LiRA-256."""
    pr = tail_metrics_at_k(
        scores, members, K_FRACS,
        vulnerable=vulnerable, gold_scores=lira_gold_scores,
    )
    return {
        "P@1%":    float(pr.iloc[0]["precision"]),
        "P@3%":    float(pr.iloc[1]["precision"]),
        "P@5%":    float(pr.iloc[2]["precision"]),
        "P@10%":   float(pr.iloc[3]["precision"]),
        "TailRho": float(pr.iloc[0]["tail_rho"]),
    }


def _read_budget_match(exp: Path) -> Dict[Tuple[str, str], int]:
    """Return {(attack, rule): R_match} from <exp>/budget_match/budget_match.csv.

    rule ∈ {"closeness", "cost_match"}.
    """
    bm_p = exp / "budget_match" / "budget_match.csv"
    if not bm_p.exists():
        raise FileNotFoundError(
            f"Missing {bm_p}. Run scripts/budget_match_sweep.py first."
        )
    df = pd.read_csv(bm_p)
    out: Dict[Tuple[str, str], int] = {}
    for atk in ["LiRA", "Attack-R", "RMIA", "Liu-MLP"]:
        for rule in ["closeness", "cost_match"]:
            sub = df[(df["attack"] == atk) & (df["rule"] == rule)]
            if sub.empty:
                continue
            out[(atk, rule)] = int(sub.iloc[0]["R_match"])
    return out


def _load_attack_scores(exp: Path, t: int, attack: str, R: int) -> np.ndarray:
    prefix = {"LiRA": "lira_refs", "Attack-R": "attackr_refs",
              "RMIA": "rmia_refs"}[attack]
    p = exp / f"model_{t}" / "attack_results" / f"{prefix}_{R}_scores.npy"
    return np.load(p).astype(np.float32)


def _load_loss_scores(exp: Path, t: int) -> np.ndarray:
    p = exp / f"model_{t}" / "attack_results" / "loss_scores.npy"
    return np.load(p).astype(np.float32)


def _load_liu_mlp(exp: Path, t: int, R: int) -> np.ndarray:
    p = exp / f"model_{t}" / f"liu_mlp_scores_R{R}.npy"
    return np.load(p).astype(np.float32)


def _load_lt_iqr(exp: Path, t: int) -> np.ndarray:
    p = exp / f"model_{t}" / "lt_iqr_scores.npy"
    return np.load(p).astype(np.float32)


def _has_rare_refs(exp: Path, t: int, M: int) -> bool:
    target_dir = exp / f"model_{int(t)}"
    return all(any(target_dir.glob(f"ref_losses_E*_*_ref{r}.npy")) for r in range(int(M)))


def _load_rare(exp: Path, t: int, M: int) -> np.ndarray:
    return load_rare_locked_scores(exp, t, num_refs=int(M), q=0.50)


def _ci95(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(arr.size))


def _fmt(val: float, ci: float, pct: bool = False) -> str:
    if not np.isfinite(val):
        return "---"
    if pct:
        return f"{val*100:.2f} \\pm {ci*100:.2f}" if np.isfinite(ci) else f"{val*100:.2f}"
    return f"{val:.4f} \\pm {ci:.4f}" if np.isfinite(ci) else f"{val:.4f}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--experiment-dir", required=True, type=str)
    ap.add_argument("--targets", type=int, nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--fpr", type=float, default=FPR_DEFAULT,
                    help="FPR at which the gold vulnerable set is defined "
                         "(default 1e-4). X values are read from budget_match.csv "
                         "irrespective of this flag.")
    args = ap.parse_args()
    fpr = float(args.fpr)
    fpr_tag = ("fpr1e-4" if abs(fpr - 1e-4) < 1e-9 else
               "fpr1e-3" if abs(fpr - 1e-3) < 1e-9 else
               f"fpr{fpr:.0e}")

    exp = Path(args.experiment_dir).resolve()
    if not exp.exists():
        print(f"[error] experiment dir not found: {exp}", file=sys.stderr)
        sys.exit(1)

    targets = list(args.targets)
    bm = _read_budget_match(exp)
    print(f"experiment_dir = {exp}")
    print(f"targets        = {targets}")
    for atk in ["LiRA", "Attack-R", "RMIA", "Liu-MLP"]:
        cl = bm.get((atk, "closeness"))
        ct = bm.get((atk, "cost_match"))
        print(f"  {atk:<8s} closeness R={cl}  cost-match R={ct}")

    rows = []
    method_order: List[str] = []
    method_R: Dict[str, int] = {}     # R per method label (-1 if N/A)
    method_cost: Dict[str, float] = {}

    def add(label: str, R: int, cost_s: float, getter: Callable):
        method_order.append(label)
        method_R[label] = R
        method_cost[label] = cost_s
        for t in targets:
            scores_full, members, vulnerable = _load_lira_gold(exp, t, fpr)
            scores = getter(t, scores_full)
            m = _per_target_metrics(scores, members, vulnerable, scores_full)
            for k, v in m.items():
                rows.append(dict(method=label, target=t, metric=k, value=v))

    # 1. LiRA gold
    add(f"LiRA (M=256, gold)", 256, LIRA_M256_COST,
        lambda t, sg: sg)

    # 2-3. LiRA closeness + cost-match
    add(f"LiRA (M={bm[('LiRA','closeness')]}, closeness)",
        bm[('LiRA','closeness')], _cost("LiRA", bm[('LiRA','closeness')]),
        lambda t, sg, M=bm[('LiRA','closeness')]: _load_attack_scores(exp, t, "LiRA", M))
    add(f"LiRA (M={bm[('LiRA','cost_match')]}, cost-match)",
        bm[('LiRA','cost_match')], _cost("LiRA", bm[('LiRA','cost_match')]),
        lambda t, sg, M=bm[('LiRA','cost_match')]: _load_attack_scores(exp, t, "LiRA", M))

    # 4-5. RMIA
    add(f"RMIA (M={bm[('RMIA','closeness')]}, closeness)",
        bm[('RMIA','closeness')], _cost("RMIA", bm[('RMIA','closeness')]),
        lambda t, sg, M=bm[('RMIA','closeness')]: _load_attack_scores(exp, t, "RMIA", M))
    add(f"RMIA (M={bm[('RMIA','cost_match')]}, cost-match)",
        bm[('RMIA','cost_match')], _cost("RMIA", bm[('RMIA','cost_match')]),
        lambda t, sg, M=bm[('RMIA','cost_match')]: _load_attack_scores(exp, t, "RMIA", M))

    # 6-7. Attack-R
    add(f"Attack-R (M={bm[('Attack-R','closeness')]}, closeness)",
        bm[('Attack-R','closeness')], _cost("Attack-R", bm[('Attack-R','closeness')]),
        lambda t, sg, M=bm[('Attack-R','closeness')]: _load_attack_scores(exp, t, "Attack-R", M))
    add(f"Attack-R (M={bm[('Attack-R','cost_match')]}, cost-match)",
        bm[('Attack-R','cost_match')], _cost("Attack-R", bm[('Attack-R','cost_match')]),
        lambda t, sg, M=bm[('Attack-R','cost_match')]: _load_attack_scores(exp, t, "Attack-R", M))

    # 8-9. Liu-MLP
    add(f"Liu-MLP (M={bm[('Liu-MLP','closeness')]}, closeness)",
        bm[('Liu-MLP','closeness')], _cost("Liu-MLP", bm[('Liu-MLP','closeness')]),
        lambda t, sg, M=bm[('Liu-MLP','closeness')]: _load_liu_mlp(exp, t, M))
    add(f"Liu-MLP (M={bm[('Liu-MLP','cost_match')]}, cost-match)",
        bm[('Liu-MLP','cost_match')], _cost("Liu-MLP", bm[('Liu-MLP','cost_match')]),
        lambda t, sg, M=bm[('Liu-MLP','cost_match')]: _load_liu_mlp(exp, t, M))

    # 10. LT-IQR (target-only, single row)
    add("LT-IQR", -1, LT_IQR_COST,
        lambda t, sg: _load_lt_iqr(exp, t))

    # 11. LOSS (target-only, single row)
    add("LOSS", -1, LOSS_COST,
        lambda t, sg: _load_loss_scores(exp, t))

    # 12-14. RARE = z_M(LT-BTM-R) + z_M(LT-IQR), M in {1, 2, 4}
    add("RARE (M=1)", 1, RARE_M1_COST,
        lambda t, sg: _load_rare(exp, t, 1))
    add("RARE (M=2)", 2, RARE_M2_COST,
        lambda t, sg: _load_rare(exp, t, 2))
    if _has_rare_refs(exp, targets[0], 4):
        add("RARE (M=4)", 4, RARE_M4_COST,
            lambda t, sg: _load_rare(exp, t, 4))

    per_target = pd.DataFrame(rows)
    out_dir = exp / "main_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_target.to_csv(out_dir / f"per_target_{fpr_tag}.csv", index=False)

    # Aggregate
    agg_rows = []
    for method in method_order:
        sub = per_target[per_target["method"] == method]
        for metric in METRIC_ORDER:
            if metric == "Cost(s)":
                agg_rows.append(dict(method=method, metric=metric,
                                     mean=method_cost[method], ci95=float("nan")))
                continue
            vals = sub[sub["metric"] == metric]["value"].to_numpy()
            mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            ci = _ci95(vals)
            agg_rows.append(dict(method=method, metric=metric, mean=mean, ci95=ci))
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / f"main_table_{fpr_tag}.csv", index=False)

    # Wide form
    wide_rows = []
    for method in method_order:
        row = {"method": method}
        for metric in METRIC_ORDER:
            r = agg[(agg["method"] == method) & (agg["metric"] == metric)].iloc[0]
            row[f"{metric}_mean"] = r["mean"]
            row[f"{metric}_ci95"] = r["ci95"]
        wide_rows.append(row)
    pd.DataFrame(wide_rows).to_csv(out_dir / f"main_table_wide_{fpr_tag}.csv", index=False)

    # LaTeX
    bench = exp.parent.parent.name + "/" + exp.parent.name
    bench_escaped = bench.replace("_", r"\_")
    tex_lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + (
            r"Main results on " + bench_escaped + r". "
            r"Mean $\pm$ 95\% CI over 10 targets. Gold $\mathcal{V}$ = LiRA-256 "
            r"hits at FPR=$" + f"{fpr:.0e}".replace("e-0", "e-") + r"$ $\cap$ members. "
            r"Closeness X = argmin$_R |P@5\%_R - P@5\%_{\text{RARE M=1}}|$. "
            r"Cost-match X = argmin$_R |\text{cost}(R) - 168\text{s}|$. "
            r"Cost from project_aere_costs.md (no amortization)."
        ) + r"}",
        r"\label{tab:main-" + bench.replace('/', '-').replace('_', '-') + r"}",
        r"\begin{tabular}{l" + "c" * len(METRIC_ORDER) + r"}",
        r"\toprule",
        " & ".join(["Method"] + [METRIC_LATEX[m] for m in METRIC_ORDER]) + r" \\",
        r"\midrule",
    ]
    for method in method_order:
        cells = [method.replace("_", r"\_").replace("%", r"\%")]
        for metric in METRIC_ORDER:
            r = agg[(agg["method"] == method) & (agg["metric"] == metric)].iloc[0]
            if metric == "Cost(s)":
                v = r["mean"]
                cells.append(f"{v:.0f}" if np.isfinite(v) else "---")
            else:
                cells.append("$" + _fmt(r["mean"], r["ci95"]) + "$")
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / f"main_table_{fpr_tag}.tex").write_text("\n".join(tex_lines) + "\n")

    # Console pretty-print (single table, six columns)
    print("\n" + "=" * 130)
    print(f"  MAIN TABLE (trade-off): {bench} — gold V at FPR={fpr:.0e}")
    print("=" * 130)
    col_widths = [40] + [18] * len(METRIC_ORDER)
    print("".join(f"{{:<{w}}}".format(s) for s, w in zip(["Method"] + METRIC_ORDER, col_widths)))
    print("-" * sum(col_widths))
    for method in method_order:
        cells = [method]
        for metric in METRIC_ORDER:
            r = agg[(agg["method"] == method) & (agg["metric"] == metric)].iloc[0]
            if metric == "Cost(s)":
                v = r["mean"]
                cells.append(f"{v:.0f}" if np.isfinite(v) else "---")
            else:
                cells.append(_fmt(r["mean"], r["ci95"]).replace(r" \pm ", "±"))
        print("".join(f"{{:<{w}}}".format(s) for s, w in zip(cells, col_widths)))
    print("=" * 130)
    print(f"\nWrote:")
    print(f"  {out_dir}/per_target_{fpr_tag}.csv")
    print(f"  {out_dir}/main_table_{fpr_tag}.csv")
    print(f"  {out_dir}/main_table_wide_{fpr_tag}.csv")
    print(f"  {out_dir}/main_table_{fpr_tag}.tex")


if __name__ == "__main__":
    main()
