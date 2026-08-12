"""Generate P@5% versus reference-budget Pareto figure."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rare.analysis.utils import load_rare_locked_scores, tail_metrics_at_k


DEFAULT_BENCHMARKS = {
    "CIFAR10": ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140",
    "CIFAR100": ROOT / "experiments/cifar100/wrn28-2/2025-12-25_0121",
    "CINIC10": ROOT / "experiments/cinic10/wrn28-2/2026-05-11_0215",
}
DEFAULT_TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
DEFAULT_OUTPUT = ROOT / "paper_tabs_figs/figures/fig_pareto.pdf"
ATTACK_BUDGETS = {
    "LiRA": [2, 4, 8, 16, 32, 64, 128, 256],
    "RMIA": [2, 4, 8, 16, 32, 64, 128, 256],
    "Attack-R": [2, 4, 8, 16, 32, 64, 128],
}
ATTACK_PREFIX = {"LiRA": "lira_refs", "RMIA": "rmia_refs", "Attack-R": "attackr_refs"}


def _load_attack_scores(exp: Path, target: int, attack: str, refs: int) -> np.ndarray:
    prefix = ATTACK_PREFIX[attack]
    return np.load(
        exp / f"model_{int(target)}" / "attack_results" / f"{prefix}_{int(refs)}_scores.npy"
    ).astype(np.float32)


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


def _precision_at_frac(
    scores: np.ndarray,
    members: np.ndarray,
    vulnerable: np.ndarray,
    frac: float,
) -> float:
    """Precision@k over a budget of ceil(k% * n_members), no |V| truncation.

    Delegates to the shared metric so this figure cannot drift from
    tab_main.py / tab_union.py.
    """
    pr = tail_metrics_at_k(
        scores, members, [float(frac)],
        vulnerable=vulnerable, cap_k_at_gold=False, k_rounding="ceil",
    )
    return float(pr.iloc[0]["precision"])


def collect_rows(
    benchmarks: Dict[str, Path],
    targets: Iterable[int],
    *,
    fpr: float,
    q: float,
    k_frac: float,
) -> pd.DataFrame:
    rows: List[dict] = []
    for data_name, exp in benchmarks.items():
        for target in targets:
            members, vulnerable = _load_lira_gold(exp, int(target), fpr)
            rare_scores = load_rare_locked_scores(exp, int(target), num_refs=1, q=q)
            lt_mlp_scores = np.load(
                exp / f"model_{int(target)}" / "liu_mlp_scores_R1.npy"
            ).astype(np.float32)
            for method, scores in [("RARE", rare_scores), ("LT-MLP", lt_mlp_scores)]:
                rows.append(
                    {
                        "data": data_name,
                        "target": int(target),
                        "method": method,
                        "M": 1,
                        "p_at_5": _precision_at_frac(
                            scores, members, vulnerable, k_frac
                        ),
                    }
                )
            for attack, budgets in ATTACK_BUDGETS.items():
                for refs in budgets:
                    scores = _load_attack_scores(exp, int(target), attack, int(refs))
                    rows.append(
                        {
                            "data": data_name,
                            "target": int(target),
                            "method": attack,
                            "M": int(refs),
                            "p_at_5": _precision_at_frac(
                                scores, members, vulnerable, k_frac
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["data", "method", "M"], sort=False)["p_at_5"]
        .agg(["mean", "std"])
        .reset_index()
    )


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    style = {
        "LiRA": dict(color="#0072B2", marker="o", linestyle="-"),
        "RMIA": dict(color="#E69F00", marker="s", linestyle="-"),
        "Attack-R": dict(color="#009E73", marker="D", linestyle="-"),
    }

    datasets = ["CIFAR10", "CIFAR100", "CINIC10"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), sharey=False)
    ylims = {"CIFAR10": (20, 102), "CIFAR100": (35, 102), "CINIC10": (30, 102)}
    for ax, data_name in zip(axes, datasets):
        subset = summary[summary["data"] == data_name]
        rare = float(
            subset[(subset["method"] == "RARE") & (subset["M"] == 1)]["mean"].iloc[0]
        ) * 100.0
        lt_mlp = float(
            subset[(subset["method"] == "LT-MLP") & (subset["M"] == 1)]["mean"].iloc[0]
        ) * 100.0
        for method in ["LiRA", "RMIA", "Attack-R"]:
            row = subset[subset["method"] == method].sort_values("M")
            ax.plot(
                row["M"].to_numpy(dtype=float),
                row["mean"].to_numpy(dtype=float) * 100.0,
                linewidth=1.4,
                markersize=4.0,
                label=method,
                **style[method],
            )
        ax.axhline(
            rare,
            color="#D55E00",
            linestyle=(0, (4, 2)),
            linewidth=1.4,
            label="RARE (M=1)",
        )
        ax.axhline(
            lt_mlp,
            color="#CC79A7",
            linestyle=(0, (1, 1.5)),
            linewidth=1.2,
            label="LT-MLP (M=1)",
        )
        ax.set_xscale("log", base=2)
        ax.set_xlim(1.0, 290)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
        ax.set_ylim(*ylims[data_name])
        ax.set_title(data_name, fontweight="bold", pad=4)
        ax.set_xlabel(r"# reference models $M$")
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.45)
        ax.grid(True, axis="x", color="#F0F0F0", linewidth=0.35)
    axes[0].set_ylabel("P@5% (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.055),
        columnspacing=1.3,
        handlelength=1.7,
        handletextpad=0.35,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0), w_pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_tex(output_pdf: Path, tex_path: Path) -> None:
    include_path = output_pdf.name if output_pdf.parent == tex_path.parent else output_pdf
    include_s = str(include_path).replace("\\", "/")
    tex = rf"""\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{{include_s}}}
\caption{{P@5\% versus reference budget $M$ under LiRA-256 gold. \rare{{}} ($M=1$) and LT-MLP ($M=1$) are shown as horizontal lines.}}
\label{{fig:pareto}}
\end{{figure*}}
"""
    tex_path.write_text(tex, encoding="utf-8")


def _first_budget_exceeding(summary: pd.DataFrame, data_name: str, method: str) -> str:
    subset = summary[summary["data"] == data_name]
    rare = float(
        subset[(subset["method"] == "RARE") & (subset["M"] == 1)]["mean"].iloc[0]
    )
    row = subset[(subset["method"] == method) & (subset["mean"] > rare)].sort_values("M")
    if row.empty:
        return "not reached"
    return str(int(row["M"].iloc[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--fpr", type=float, default=1e-4)
    parser.add_argument("--q", type=float, default=0.50)
    parser.add_argument("--k-frac", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(
        DEFAULT_BENCHMARKS,
        args.targets,
        fpr=float(args.fpr),
        q=float(args.q),
        k_frac=float(args.k_frac),
    )
    summary = summarize_rows(rows)
    output = Path(args.output)
    rows.to_csv(output.with_suffix(".per_target.csv"), index=False)
    summary.to_csv(output.with_suffix(".summary.csv"), index=False)
    plot_summary(summary, output)
    write_tex(output, output.with_suffix(".tex"))

    for data_name in ["CIFAR10", "CIFAR100", "CINIC10"]:
        print(data_name)
        rare = float(
            summary[
                (summary["data"] == data_name)
                & (summary["method"] == "RARE")
                & (summary["M"] == 1)
            ]["mean"].iloc[0]
        )
        lt_mlp = float(
            summary[
                (summary["data"] == data_name)
                & (summary["method"] == "LT-MLP")
                & (summary["M"] == 1)
            ]["mean"].iloc[0]
        )
        print(f"  RARE P@5={rare:.4f}  LT-MLP P@5={lt_mlp:.4f}")
        for method in ["LiRA", "RMIA", "Attack-R"]:
            print(f"  {method} first > RARE: {_first_budget_exceeding(summary, data_name, method)}")
        wide = (
            summary[summary["data"] == data_name]
            .pivot(index="M", columns="method", values="mean")
            .sort_index()
        )
        print(wide.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
