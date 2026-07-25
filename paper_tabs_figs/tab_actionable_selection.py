"""Generate actionable vulnerable-set selection comparison table."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve

from rare.analysis.utils import score_avg_min_q_reference_losses


DEFAULT_BENCHMARKS = {
    "CIFAR10": ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140",
    "CIFAR100": ROOT / "experiments/cifar100/wrn28-2/2025-12-25_0121",
    "CINIC10": ROOT / "experiments/cinic10/wrn28-2/2026-05-11_0215",
}
DEFAULT_TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
DEFAULT_OUTPUT = ROOT / "paper_tabs_figs/figures/tab_actionable_selection.csv"
METHOD_ORDER = [
    "KMeans-RARE",
    "KMeans-LT-IQR",
    "LOSS",
    "LiRA (M=8)",
    "RMIA (M=8)",
    "Attack-R (M=16)",
    "LT-MLP (M=1)",
]


def _zscore_on_members(scores: np.ndarray, members: np.ndarray) -> np.ndarray:
    out = np.full(scores.shape, np.nan, dtype=np.float32)
    ok = members & np.isfinite(scores)
    vals = scores[ok].astype(np.float32, copy=False)
    if vals.size == 0:
        return out
    mu = float(vals.mean())
    sd = float(vals.std())
    out[ok] = (vals - mu) / sd if sd > 0 and np.isfinite(sd) else vals - mu
    return out


def _load_lira_gold(exp: Path, target: int, fpr: float) -> tuple[np.ndarray, np.ndarray]:
    attack_dir = exp / f"model_{int(target)}" / "attack_results"
    scores = np.load(attack_dir / "lira_refs_256_scores.npy").astype(np.float32)
    labels = np.load(attack_dir / "lira_refs_256_labels.npy").astype(bool)
    thresholds = np.load(
        attack_dir / "lira_refs_256_thresholds_by_fpr.npy",
        allow_pickle=True,
    ).item()
    vulnerable = (scores >= float(thresholds[float(fpr)])) & labels
    return labels, vulnerable


def _load_lt_iqr(exp: Path, target: int) -> np.ndarray:
    return np.load(exp / f"model_{int(target)}" / "lt_iqr_scores.npy").astype(np.float32)


def _load_lt_btm_r(exp: Path, target: int, q: float) -> np.ndarray:
    target_dir = exp / f"model_{int(target)}"
    matches = sorted(target_dir.glob("ref_losses_E*_*_ref0.npy"))
    if not matches:
        raise FileNotFoundError(f"No ref_losses_E*_*_ref0.npy in {target_dir}")
    return score_avg_min_q_reference_losses([np.load(matches[0]).astype(np.float32)], q=q)


def _kmeans_select(
    features: np.ndarray,
    members: np.ndarray,
    polarity_scores: np.ndarray,
) -> np.ndarray:
    ok = members & np.all(np.isfinite(features), axis=1) & np.isfinite(polarity_scores)
    idx = np.where(ok)[0]
    labels = KMeans(n_clusters=2, random_state=0, n_init=50).fit_predict(features[idx])
    means = [float(polarity_scores[idx][labels == c].mean()) for c in (0, 1)]
    chosen = int(np.argmax(means))
    selected = np.zeros(members.shape, dtype=bool)
    selected[idx[labels == chosen]] = True
    return selected


def _threshold_select(exp: Path, target: int, filename_prefix: str, fpr: float) -> np.ndarray:
    attack_dir = exp / f"model_{int(target)}" / "attack_results"
    scores = np.load(attack_dir / f"{filename_prefix}_scores.npy").astype(np.float32)
    thresholds = np.load(
        attack_dir / f"{filename_prefix}_thresholds_by_fpr.npy",
        allow_pickle=True,
    ).item()
    return scores >= float(thresholds[float(fpr)])


def _score_threshold_select(
    scores: np.ndarray, members: np.ndarray, fpr: float
) -> np.ndarray:
    """Threshold raw attack scores at the FPR target.

    For methods without a saved `*_thresholds_by_fpr.npy` (LT-MLP), the
    threshold is derived from the scores with the same convention as
    scripts/attacks.py `compute_metrics`: the threshold of the largest
    ROC point with FPR <= target."""
    ok = np.isfinite(scores)
    fpr_arr, _, thr = roc_curve(members[ok].astype(int), scores[ok])
    idx = np.where(fpr_arr <= float(fpr))[0]
    t = float(thr[idx[-1]]) if idx.size else float("inf")
    return ok & (scores >= t)


def _metrics(selected: np.ndarray, members: np.ndarray, vulnerable: np.ndarray) -> dict:
    selected_members = selected & members
    tp = int((selected_members & vulnerable).sum())
    n = int(selected_members.sum())
    gold_n = int(vulnerable.sum())
    precision = float(tp / n) if n else 0.0
    recall = float(tp / gold_n) if gold_n else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {
        "selected": n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "gold": gold_n,
    }


def collect_rows(
    benchmarks: Dict[str, Path],
    targets: Iterable[int],
    *,
    fpr: float,
    q: float,
) -> pd.DataFrame:
    rows: List[dict] = []
    for data_name, exp in benchmarks.items():
        for target in targets:
            members, vulnerable = _load_lira_gold(exp, int(target), fpr)
            lt_iqr = _load_lt_iqr(exp, int(target))
            lt_btm_r = _load_lt_btm_r(exp, int(target), q)
            z_iqr = _zscore_on_members(lt_iqr, members)
            z_btm = _zscore_on_members(lt_btm_r, members)
            rare = z_iqr + z_btm

            selectors = {
                "KMeans-RARE": _kmeans_select(
                    np.column_stack([z_iqr, z_btm]), members, rare
                ),
                "KMeans-LT-IQR": _kmeans_select(
                    z_iqr.reshape(-1, 1), members, z_iqr
                ),
                "LOSS": _threshold_select(exp, int(target), "loss", fpr),
                "LiRA (M=8)": _threshold_select(exp, int(target), "lira_refs_8", fpr),
                "RMIA (M=8)": _threshold_select(exp, int(target), "rmia_refs_8", fpr),
                "Attack-R (M=16)": _threshold_select(
                    exp, int(target), "attackr_refs_16", fpr
                ),
                "LT-MLP (M=1)": _score_threshold_select(
                    np.load(
                        exp / f"model_{int(target)}" / "liu_mlp_scores_R1.npy"
                    ).astype(np.float32),
                    members,
                    fpr,
                ),
            }
            for method, selected in selectors.items():
                row = {
                    "data": data_name,
                    "target": int(target),
                    "method": method,
                }
                row.update(_metrics(selected, members, vulnerable))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["data", "method"], sort=False)
        .agg(
            selected=("selected", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
        )
        .reset_index()
    )
    avg = (
        summary.groupby("method", sort=False)
        .agg(
            selected=("selected", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
        )
        .reset_index()
    )
    avg.insert(0, "data", "Avg.")
    return pd.concat([summary, avg], ignore_index=True)


def _fmt_metric(x: float) -> str:
    return f"{float(x):.3f}"


def _fmt_selected(x: float) -> str:
    return f"{float(x):.1f}"


def _ranked_metric(
    summary: pd.DataFrame,
    data_name: str,
    method: str,
    metric: str,
) -> str:
    value = float(
        summary[(summary["data"] == data_name) & (summary["method"] == method)][metric].iloc[0]
    )
    values = (
        summary[summary["data"] == data_name]
        .set_index("method")
        .loc[METHOD_ORDER, metric]
        .astype(float)
    )
    ordered = values.sort_values(ascending=False)
    best = float(ordered.iloc[0])
    second = float(ordered.iloc[1])
    out = _fmt_metric(value)
    if np.isclose(value, best, rtol=0.0, atol=5e-4):
        return rf"\textbf{{{out}}}"
    if np.isclose(value, second, rtol=0.0, atol=5e-4):
        return rf"\underline{{{out}}}"
    return out


def write_tex(summary: pd.DataFrame, tex_path: Path, gold_sizes: Dict[str, float]) -> None:
    sizes = " / ".join(f"{gold_sizes[d]:.1f}" for d in ["CIFAR10", "CIFAR100", "CINIC10"])
    lines = [
        r"% Auto-generated by paper_tabs_figs/tab_actionable_selection.py -- do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Actionable vulnerable-set selection under LiRA-256 gold at FPR$\le 10^{-4}$. KMeans-\rare{} clusters the two locked \rare{} components, $z_M(\mathrm{LT\mbox{-}IQR})$ and $z_M(\mathrm{LT\mbox{-}BTM\mbox{-}R})$, over target members and selects the higher-\rare{} cluster. "
        f"The average LiRA-256 gold-set sizes are {sizes} members on CIFAR-10 / CIFAR-100 / CINIC-10, respectively.}}",
        r"\label{tab:actionable-selection}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l cccc cccc cccc cccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Method}}",
        r"  & \multicolumn{4}{c}{\textbf{CIFAR10}}",
        r"  & \multicolumn{4}{c}{\textbf{CIFAR100}}",
        r"  & \multicolumn{4}{c}{\textbf{CINIC10}}",
        r"  & \multicolumn{4}{c}{\textbf{Avg.}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-17}",
        r"& \textbf{$|S|$} & \textbf{P} & \textbf{R} & \textbf{F1}",
        r"& \textbf{$|S|$} & \textbf{P} & \textbf{R} & \textbf{F1}",
        r"& \textbf{$|S|$} & \textbf{P} & \textbf{R} & \textbf{F1}",
        r"& \textbf{$|S|$} & \textbf{P} & \textbf{R} & \textbf{F1} \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        label = method.replace("KMeans-RARE", r"KMeans-\rare{}")
        cells = [label]
        for data_name in ["CIFAR10", "CIFAR100", "CINIC10", "Avg."]:
            row = summary[(summary["data"] == data_name) & (summary["method"] == method)].iloc[0]
            cells.extend(
                [
                    _fmt_selected(row["selected"]),
                    _ranked_metric(summary, data_name, method, "precision"),
                    _ranked_metric(summary, data_name, method, "recall"),
                    _ranked_metric(summary, data_name, method, "f1"),
                ]
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--fpr", type=float, default=1e-4)
    parser.add_argument("--q", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(DEFAULT_BENCHMARKS, args.targets, fpr=float(args.fpr), q=float(args.q))
    summary = summarize_rows(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output.with_suffix(".per_target.csv"), index=False)
    summary.to_csv(output, index=False)
    gold_sizes = rows.groupby("data")["gold"].mean().to_dict()
    write_tex(summary, output.with_suffix(".tex"), gold_sizes)
    print("mean |V| per dataset:",
          {d: round(v, 1) for d, v in gold_sizes.items()})

    for data_name in ["CIFAR10", "CIFAR100", "CINIC10", "Avg."]:
        print(f"\nAverage over {data_name} targets")
        sub = summary[summary["data"] == data_name].set_index("method").loc[METHOD_ORDER]
        for method, row in sub.iterrows():
            print(
                f"  {method:<18} n={row['selected']:7.1f} "
                f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}"
            )


if __name__ == "__main__":
    main()
