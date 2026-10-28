"""Generate P@k% versus ranking-depth figure."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_tabs_figs._tex import figure_env  # noqa: E402

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
DEFAULT_OUTPUT = ROOT / "paper_tabs_figs/figures/fig_p_vs_k.pdf"
K_FRACS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
METHODS = [
    "LiRA (M=8)",
    "RMIA (M=8)",
    "Attack-R (M=16)",
    "LT-MLP (M=1)",
    "LT-IQR (M=0)",
    "LOSS (M=0)",
    "RARE (M=1)",
]


def _load_attack(exp: Path, target: int, prefix: str, refs: int) -> np.ndarray:
    return np.load(
        exp / f"model_{int(target)}" / "attack_results" / f"{prefix}_{int(refs)}_scores.npy"
    ).astype(np.float32)


def _load_lira_gold(exp: Path, target: int, fpr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    attack_dir = exp / f"model_{int(target)}" / "attack_results"
    scores = np.load(attack_dir / "lira_refs_256_scores.npy").astype(np.float32)
    labels = np.load(attack_dir / "lira_refs_256_labels.npy").astype(bool)
    thresholds = np.load(
        attack_dir / "lira_refs_256_thresholds_by_fpr.npy",
        allow_pickle=True,
    ).item()
    vulnerable = (scores >= float(thresholds[float(fpr)])) & labels
    return scores, labels, vulnerable


def _method_loaders(q: float) -> Dict[str, Callable[[Path, int], np.ndarray]]:
    return {
        "LiRA (M=8)": lambda exp, target: _load_attack(exp, target, "lira_refs", 8),
        "RMIA (M=8)": lambda exp, target: _load_attack(exp, target, "rmia_refs", 8),
        "Attack-R (M=16)": lambda exp, target: _load_attack(exp, target, "attackr_refs", 16),
        "LT-MLP (M=1)": lambda exp, target: np.load(
            exp / f"model_{int(target)}" / "liu_mlp_scores_R1.npy"
        ).astype(np.float32),
        "LT-IQR (M=0)": lambda exp, target: np.load(
            exp / f"model_{int(target)}" / "lt_iqr_scores.npy"
        ).astype(np.float32),
        "LOSS (M=0)": lambda exp, target: np.load(
            exp / f"model_{int(target)}" / "attack_results" / "loss_scores.npy"
        ).astype(np.float32),
        "RARE (M=1)": lambda exp, target: load_rare_locked_scores(
            exp, int(target), num_refs=1, q=q
        ),
    }


def collect_rows(
    benchmarks: Dict[str, Path],
    targets: Iterable[int],
    *,
    fpr: float,
    q: float,
) -> pd.DataFrame:
    loaders = _method_loaders(q)
    rows: List[dict] = []
    for data_name, exp in benchmarks.items():
        for target in targets:
            gold_scores, members, vulnerable = _load_lira_gold(exp, int(target), fpr)
            for method in METHODS:
                scores = loaders[method](exp, int(target))
                metrics = tail_metrics_at_k(
                    scores,
                    members,
                    K_FRACS,
                    vulnerable=vulnerable,
                    gold_scores=gold_scores,
                )
                for k_frac, (_, metric_row) in zip(K_FRACS, metrics.iterrows()):
                    rows.append(
                        {
                            "data": data_name,
                            "target": int(target),
                            "method": method,
                            "k_frac": float(k_frac),
                            "k_pct": float(k_frac * 100.0),
                            "precision": float(metric_row["precision"]),
                            "k_samples": int(metric_row["k_samples"]),
                            "gold_size": int(metric_row["gold_n"]),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["data", "method", "k_frac", "k_pct"], sort=False)
        .agg(
            mean=("precision", "mean"),
            std=("precision", "std"),
            k_samples=("k_samples", "mean"),
            gold_size=("gold_size", "mean"),
        )
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
        "LiRA (M=8)": dict(color="#0072B2", marker="o", linestyle="-"),
        "RMIA (M=8)": dict(color="#E69F00", marker="s", linestyle="-"),
        "Attack-R (M=16)": dict(color="#009E73", marker="D", linestyle="-"),
        "LT-MLP (M=1)": dict(color="#CC79A7", marker="^", linestyle="-"),
        "LT-IQR (M=0)": dict(color="#56B4E9", marker="v", linestyle="--"),
        "LOSS (M=0)": dict(color="#8A8A8A", marker="x", linestyle=":"),
        "RARE (M=1)": dict(color="#D55E00", marker="*", linestyle="-"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), sharey=True)
    for ax, data_name in zip(axes, ["CIFAR10", "CIFAR100", "CINIC10"]):
        subset = summary[summary["data"] == data_name]
        for method in METHODS:
            row = subset[subset["method"] == method].sort_values("k_frac")
            ax.plot(
                row["k_pct"].to_numpy(dtype=float),
                row["mean"].to_numpy(dtype=float) * 100.0,
                linewidth=1.3,
                markersize=3.8 if method != "RARE (M=1)" else 5.2,
                label=method,
                **style[method],
            )
        ax.set_xscale("log")
        ax.set_xlim(0.85, 58)
        ax.set_ylim(40, 102)
        ax.set_xticks([1, 3, 10, 30, 50])
        ax.set_xticklabels(["1", "3", "10", "30", "50"])
        ax.set_title(data_name, fontweight="bold", pad=4)
        ax.set_xlabel("Top-k%")
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.45)
        ax.grid(True, axis="x", color="#F0F0F0", linewidth=0.35)
    axes[0].set_ylabel("P@k% (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.25,
    )
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0), w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


CAPTION = r"""P@$k$\% as the ranking depth $k$ varies from $1\%$ to $50\%$.
LOSS falls below the plotted range at all depths."""


def write_tex(output_pdf: Path, tex_path: Path) -> None:
    tex = figure_env(output_pdf.name, CAPTION, "fig:p_vs_k")
    tex_path.write_text(tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--fpr", type=float, default=1e-4)
    parser.add_argument("--q", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(
        DEFAULT_BENCHMARKS,
        args.targets,
        fpr=float(args.fpr),
        q=float(args.q),
    )
    summary = summarize_rows(rows)
    output = Path(args.output)
    rows.to_csv(output.with_suffix(".per_target.csv"), index=False)
    summary.to_csv(output.with_suffix(".summary.csv"), index=False)
    plot_summary(summary, output)
    write_tex(output, output.with_suffix(".tex"))

    for data_name in ["CIFAR10", "CIFAR100", "CINIC10"]:
        print(data_name)
        wide = (
            summary[
                (summary["data"] == data_name)
                & (summary["k_frac"].isin([0.01, 0.05, 0.10, 0.50]))
            ]
            .pivot(index="method", columns="k_pct", values="mean")
            .loc[METHODS]
        )
        print(wide.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
