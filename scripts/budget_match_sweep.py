"""Budget-match sweep with cost model.

For each shadow attack (LiRA, Attack-R, RMIA, Liu-MLP) and each R, compute
capped Precision@k=k_frac at FPR=1e-4 per target, aggregate across targets,
and determine TWO matching points vs RARE M=1:

  closeness  : argmin_R |mean_precision(R) - precision_RARE_R1|
  cost_match : argmin_R |cost(R) - cost_RARE_R1|

Cost model (per project_aere_costs.md, no amortization):
  LiRA     R : R * 120 + R * 1.5 + 1.5 + 1.2     = R*121.5 + 2.7
  Attack-R R : R * 120 + R * 1.5 + 1.5 + 4.0     = R*121.5 + 5.5
  RMIA     R : R * 120 + R * 1.5 + 1.5 + 10.0    = R*121.5 + 11.5
  Liu-MLP  R : R * 144 + 5                       (each shadow needs trace)
  LOSS       : ~1.5                              (target logit only)
  RARE M=1   : 168   = 24 + 1*144
  RARE M=2   : 312   = 24 + 2*144

Outputs (under <experiment_dir>/budget_match/):
  precision_curves.csv  -- per (target, attack, R) precision
  precision_per_R.csv   -- mean ± CI95 across targets per (attack, R)
  budget_match.csv      -- closeness + cost_match X per attack
"""
import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rare.analysis.utils import load_rare_locked_scores, tail_metrics_at_k


# Cost model — per project_aere_costs.md, no amortization.
# RARE cost: 24s target-trace overhead + M*144s reference retrains.
def _rare_cost(M: int) -> float:
    return 24.0 + M * 144.0

RARE_COST = {1: _rare_cost(1), 2: _rare_cost(2), 4: _rare_cost(4)}


def _cost(method: str, M: int) -> float:
    if method == "LiRA":     return M * 121.5 + 2.7
    if method == "Attack-R": return M * 121.5 + 5.5
    if method == "RMIA":     return M * 121.5 + 11.5
    if method == "Liu-MLP":  return M * 144.0 + 5.0
    raise ValueError(f"unknown method: {method}")


def _load_lira_gold(exp: Path, t: int, fpr: float):
    ar = exp / f"model_{t}" / "attack_results"
    scores = np.load(ar / "lira_refs_256_scores.npy").astype(np.float32)
    labels = np.load(ar / "lira_refs_256_labels.npy").astype(bool)
    thr_dict = np.load(
        ar / "lira_refs_256_thresholds_by_fpr.npy", allow_pickle=True
    ).item()
    if fpr not in thr_dict:
        raise KeyError(
            f"FPR={fpr} not in LiRA thresholds for target {t}: "
            f"keys={sorted(thr_dict.keys())}"
        )
    thr = float(thr_dict[fpr])
    vulnerable = (scores >= thr) & labels
    return labels, vulnerable


def _precision_at_k(scores: np.ndarray, members: np.ndarray,
                    vulnerable: np.ndarray, k_frac: float) -> float:
    pr = tail_metrics_at_k(
        scores, members, [k_frac], vulnerable=vulnerable,
    )
    return float(pr.iloc[0]["precision"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-dir", required=True, type=str)
    ap.add_argument("--targets", type=int, nargs="*",
                    default=[0, 25, 50, 75, 100, 125, 150, 175, 200, 250])
    ap.add_argument("--fpr", type=float, default=1e-4)
    ap.add_argument("--k-frac", type=float, default=0.01)
    ap.add_argument("--rare-m", "--rare-r", type=int, default=1, dest="rare_m",
                    help="RARE M value to match (default 1). Alias --rare-r kept for back-compat.")
    ap.add_argument("--lira-grid", type=int, nargs="+",
                    default=[2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256])
    ap.add_argument("--attackr-grid", type=int, nargs="+",
                    default=[2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128])
    ap.add_argument("--rmia-grid", type=int, nargs="+",
                    default=[2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256])
    ap.add_argument("--liu-grid", type=int, nargs="+",
                    default=[1, 2, 4])
    args = ap.parse_args()

    exp = Path(args.experiment_dir).resolve()
    if not exp.exists():
        print(f"[error] experiment-dir not found: {exp}", file=sys.stderr)
        sys.exit(1)

    print(f"experiment_dir = {exp}")
    print(f"targets        = {args.targets}")
    print(f"FPR threshold  = {args.fpr:.0e}")
    print(f"k_frac         = {args.k_frac:.2%}")
    print(f"RARE M         = {args.rare_m}  score = z_M(LT-BTM-R) + z_M(LT-IQR)")

    rows = []
    for t in args.targets:
        try:
            members, vulnerable = _load_lira_gold(exp, t, args.fpr)
        except FileNotFoundError as e:
            print(f"[error] target {t}: {e}", file=sys.stderr)
            sys.exit(1)
        n_v = int(vulnerable.sum())

        try:
            rare_scores = load_rare_locked_scores(exp, t, num_refs=args.rare_m, q=0.50)
        except (FileNotFoundError, ValueError) as e:
            print(f"[error] target {t}: cannot compute RARE M={args.rare_m}: {e}", file=sys.stderr)
            sys.exit(1)
        p_rare = _precision_at_k(rare_scores, members, vulnerable, args.k_frac)
        rows.append(dict(target=t, attack="RARE", R=args.rare_m,
                         head="z_M(LT-BTM-R)+z_M(LT-IQR)", precision=p_rare, gold_n=n_v))

        ar = exp / f"model_{t}" / "attack_results"
        for attack_name, prefix, grid in [
            ("LiRA", "lira_refs", args.lira_grid),
            ("Attack-R", "attackr_refs", args.attackr_grid),
            ("RMIA", "rmia_refs", args.rmia_grid),
        ]:
            for R in grid:
                sp = ar / f"{prefix}_{R}_scores.npy"
                if not sp.exists():
                    rows.append(dict(target=t, attack=attack_name, R=R,
                                     head="-", precision=float("nan"),
                                     gold_n=n_v))
                    continue
                scores = np.load(sp).astype(np.float32)
                p = _precision_at_k(scores, members, vulnerable, args.k_frac)
                rows.append(dict(target=t, attack=attack_name, R=R,
                                 head="-", precision=p, gold_n=n_v))

        # Liu-MLP (supervised LT baseline) — score file lives in model_<t>/,
        # not attack_results/. R = number of labelled shadow targets.
        for R in args.liu_grid:
            lp = exp / f"model_{t}" / f"liu_mlp_scores_R{R}.npy"
            if not lp.exists():
                rows.append(dict(target=t, attack="Liu-MLP", R=R,
                                 head="-", precision=float("nan"), gold_n=n_v))
                continue
            scores = np.load(lp).astype(np.float32)
            p = _precision_at_k(scores, members, vulnerable, args.k_frac)
            rows.append(dict(target=t, attack="Liu-MLP", R=R,
                             head="-", precision=p, gold_n=n_v))

    df = pd.DataFrame(rows)
    out_dir = exp / "budget_match"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "precision_curves.csv", index=False)

    agg = (
        df.groupby(["attack", "R"], as_index=False)
        .agg(precision_mean=("precision", "mean"),
             precision_std=("precision", "std"),
             n=("precision", "count"),
             gold_n_mean=("gold_n", "mean"))
    )
    agg["precision_ci95"] = 1.96 * agg["precision_std"] / np.sqrt(agg["n"])
    agg = agg.sort_values(["attack", "R"]).reset_index(drop=True)
    agg.to_csv(out_dir / "precision_per_R.csv", index=False)

    rare_row = agg[(agg["attack"] == "RARE") & (agg["R"] == args.rare_m)]
    if rare_row.empty:
        print("[error] RARE row missing from aggregation", file=sys.stderr)
        sys.exit(1)
    rare_target = float(rare_row["precision_mean"].iloc[0])
    rare_ci = float(rare_row["precision_ci95"].iloc[0])
    print(f"\n=== Budget-match summary: {exp.name} ===")
    print(f"RARE M={args.rare_m} (z_M(LT-BTM-R)+z_M(LT-IQR)) mean Precision@k="
          f"{args.k_frac:.0%} (FPR={args.fpr:.0e}): "
          f"{rare_target:.4f} ± {rare_ci:.4f}")

    summary_rows = [dict(
        attack="RARE", rule="-", R_match=args.rare_m,
        mean_precision=rare_target, ci95=rare_ci, gap=0.0,
        cost_seconds=RARE_COST.get(args.rare_m, _rare_cost(args.rare_m)),
        relative_cost=1.0,
    )]
    print(f"RARE cost = {RARE_COST.get(args.rare_m, _rare_cost(args.rare_m)):.0f}s "
          f"(reference for cost-match)\n")
    for atk in ["LiRA", "Attack-R", "RMIA", "Liu-MLP"]:
        sub = agg[agg["attack"] == atk].sort_values("R").reset_index(drop=True)
        if sub.empty:
            print(f"  {atk:<8s}  (no scores in grid)")
            continue
        sub_g = sub.assign(gap=sub["precision_mean"] - rare_target)
        sub_g["cost"] = sub_g["R"].astype(int).map(lambda r: _cost(atk, int(r)))

        # closeness — R with smallest |precision − RARE|
        cl_idx = sub_g["gap"].abs().idxmin()
        cl_row = sub_g.loc[cl_idx]
        cl_R, cl_P, cl_CI, cl_g, cl_cost = (
            int(cl_row["R"]), float(cl_row["precision_mean"]),
            float(cl_row["precision_ci95"]), float(cl_row["gap"]),
            float(cl_row["cost"]),
        )

        # cost-match — R with smallest |cost - RARE_cost|
        rare_cost = RARE_COST.get(args.rare_m, _rare_cost(args.rare_m))
        ct_idx = (sub_g["cost"] - rare_cost).abs().idxmin()
        ct_row = sub_g.loc[ct_idx]
        ct_R, ct_P, ct_CI, ct_g, ct_cost = (
            int(ct_row["R"]), float(ct_row["precision_mean"]),
            float(ct_row["precision_ci95"]), float(ct_row["gap"]),
            float(ct_row["cost"]),
        )

        print(f"  {atk:<8s}  closeness   R={cl_R:>3d}  P={cl_P:.4f}±{cl_CI:.4f}  gap={cl_g:+.4f}  cost={cl_cost:>6.0f}s ({cl_cost/rare_cost:.2f}× RARE)")
        print(f"  {'':<8s}  cost-match  R={ct_R:>3d}  P={ct_P:.4f}±{ct_CI:.4f}  gap={ct_g:+.4f}  cost={ct_cost:>6.0f}s ({ct_cost/rare_cost:.2f}× RARE)")

        summary_rows.append(dict(attack=atk, rule="closeness", R_match=cl_R,
                                 mean_precision=cl_P, ci95=cl_CI, gap=cl_g,
                                 cost_seconds=cl_cost, relative_cost=cl_cost / rare_cost))
        summary_rows.append(dict(attack=atk, rule="cost_match", R_match=ct_R,
                                 mean_precision=ct_P, ci95=ct_CI, gap=ct_g,
                                 cost_seconds=ct_cost, relative_cost=ct_cost / rare_cost))

    pd.DataFrame(summary_rows).to_csv(out_dir / "budget_match.csv", index=False)
    print(f"\nWrote: {out_dir}/precision_curves.csv")
    print(f"       {out_dir}/precision_per_R.csv")
    print(f"       {out_dir}/budget_match.csv")


if __name__ == "__main__":
    main()
