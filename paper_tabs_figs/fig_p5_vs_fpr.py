"""Generate P@5% versus LiRA-256 gold-set FPR figure."""

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
DEFAULT_OUTPUT = ROOT / "paper_tabs_figs/figures/fig_p5_vs_fpr.pdf"
FPRS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
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


def _load_lira_gold_inputs(exp: Path, target: int) -> tuple[np.ndarray, np.ndarray, dict]:
    attack_dir = exp / f"model_{int(target)}" / "attack_results"
    scores = np.load(attack_dir / "lira_refs_256_scores.npy").astype(np.float32)
    labels = np.load(attack_dir / "lira_refs_256_labels.npy").astype(bool)
    thresholds = np.load(
        attack_dir / "lira_refs_256_thresholds_by_fpr.npy",
        allow_pickle=True,
    ).item()
    return scores, labels, thresholds


def _threshold_for_fpr(
    gold_scores: np.ndarray,
    labels: np.ndarray,
    thresholds: dict,
    fpr: float,
) -> float:
    fpr = float(fpr)
    if fpr in thresholds:
        return float(thresholds[fpr])
    nonmember_scores = gold_scores[~labels]
    if fpr <= 0:
        return float(np.nextafter(np.max(nonmember_scores), np.inf))
    if fpr >= 1:
        return float(-np.inf)
    allowed = int(np.floor(fpr * len(nonmember_scores)))
    if allowed <= 0:
        return float(np.nextafter(np.max(nonmember_scores), np.inf))
    desc = np.sort(nonmember_scores)[::-1]
    return float(desc[min(allowed - 1, len(desc) - 1)])


def _precision_at_5(
    scores: np.ndarray,
    members: np.ndarray,
    vulnerable: np.ndarray,
) -> float:
    """P@5% over a budget of ceil(5% * n_members), no |V| truncation.

    Delegates to the shared metric so this figure cannot drift from
    tab_main.py / fig_pareto.py.
    """
    pr = tail_metrics_at_k(
        scores, members, [0.05],
        vulnerable=vulnerable,
    )
    return float(pr.iloc[0]["precision"])


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
    fprs: Iterable[float],
    q: float,
) -> pd.DataFrame:
    loaders = _method_loaders(q)
    rows: List[dict] = []
    for data_name, exp in benchmarks.items():
        for target in targets:
            gold_scores, members, thresholds = _load_lira_gold_inputs(exp, int(target))
            method_scores = {name: loader(exp, int(target)) for name, loader in loaders.items()}
            for fpr in fprs:
                threshold = _threshold_for_fpr(gold_scores, members, thresholds, float(fpr))
                vulnerable = (gold_scores >= threshold) & members
                for method in METHODS:
                    rows.append(
                        {
                            "data": data_name,
                            "target": int(target),
                            "method": method,
                            "fpr": float(fpr),
                            "p_at_5": _precision_at_5(
                                method_scores[method], members, vulnerable
                            ),
                            "gold_size": int(vulnerable.sum()),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["data", "method", "fpr"], sort=False)
        .agg(mean=("p_at_5", "mean"), std=("p_at_5", "std"), gold_size=("gold_size", "mean"))
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
            row = subset[subset["method"] == method].sort_values("fpr")
            ax.plot(
                row["fpr"].to_numpy(dtype=float),
                row["mean"].to_numpy(dtype=float) * 100.0,
                linewidth=1.3,
                markersize=3.8 if method != "RARE (M=1)" else 5.2,
                label=method,
                **style[method],
            )
        ax.axvline(1e-4, color="#555555", linestyle=":", linewidth=1.0)
        ax.set_xscale("symlog", linthresh=1e-5, linscale=0.75)
        ax.set_xlim(-1e-6, 1.25)
        ax.set_ylim(0, 102)
        ax.set_xticks([0, 1e-4, 1e-2, 1.0])
        ax.set_xticklabels(["0", r"$10^{-4}$", r"$10^{-2}$", "1"])
        ax.set_title(data_name, fontweight="bold", pad=4)
        ax.set_xlabel("FPR")
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.45)
        ax.grid(True, axis="x", color="#F0F0F0", linewidth=0.35)
    axes[0].set_ylabel("P@5% (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.055),
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.25,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0), w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


CAPTION = r"""P@5\% as the LiRA-256 gold-set FPR threshold is varied. The
dotted vertical line marks $\mathrm{FPR}=10^{-4}$, the operating point
used throughout."""


def write_tex(output_pdf: Path, tex_path: Path) -> None:
    tex = figure_env(output_pdf.name, CAPTION, "fig:p5_vs_fpr")
    tex_path.write_text(tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--q", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(DEFAULT_BENCHMARKS, args.targets, fprs=FPRS, q=float(args.q))
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
                & (summary["fpr"].isin([0.0, 1e-4, 1e-2]))
            ]
            .pivot(index="method", columns="fpr", values="mean")
            .loc[METHODS]
        )
        print(wide.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
