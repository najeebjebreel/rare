"""Generate the ranking-quality RBO figure for the paper."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rare.analysis.utils import load_rare_locked_scores


DEFAULT_BENCHMARKS = {
    "CIFAR10": ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140",
    "CIFAR100": ROOT / "experiments/cifar100/wrn28-2/2025-12-25_0121",
    "CINIC10": ROOT / "experiments/cinic10/wrn28-2/2026-05-11_0215",
}
DEFAULT_TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
DEFAULT_OUTPUT = ROOT / "paper_tabs_figs/figures/fig_ranking_rbo.pdf"
METHODS = [
    "LiRA (M=8)",
    "RMIA (M=8)",
    "Attack-R (M=16)",
    "LT-MLP (M=1)",
    "LT-IQR (M=0)",
    "LOSS (M=0)",
    "RARE (M=1)",
]
K_FRACS = [0.01, 0.03, 0.05, 0.10, 1.00]


def _load_keep_mask(exp: Path, target: int, n: int) -> np.ndarray:
    keep = np.load(exp / "keep_indices.npy")
    mask = np.asarray(keep[int(target)] if keep.ndim == 2 else keep, dtype=bool)
    if mask.shape[0] != int(n):
        raise ValueError(f"keep mask length {mask.shape[0]} != score length {n}")
    return mask


def _load_attack(exp: Path, target: int, prefix: str, num_refs: int) -> np.ndarray:
    return np.load(
        exp / f"model_{int(target)}" / "attack_results" / f"{prefix}_{int(num_refs)}_scores.npy"
    ).astype(np.float32)


def _load_loss(exp: Path, target: int) -> np.ndarray:
    return np.load(
        exp / f"model_{int(target)}" / "attack_results" / "loss_scores.npy"
    ).astype(np.float32)


def _load_lt_iqr(exp: Path, target: int) -> np.ndarray:
    return np.load(exp / f"model_{int(target)}" / "lt_iqr_scores.npy").astype(np.float32)


def _load_lt_mlp(exp: Path, target: int) -> np.ndarray:
    return np.load(exp / f"model_{int(target)}" / "liu_mlp_scores_R1.npy").astype(np.float32)


def _rbo_ext_at_k(method_scores: np.ndarray, gold_scores: np.ndarray, k_frac: float) -> float:
    ok = np.isfinite(method_scores) & np.isfinite(gold_scores)
    method_scores = method_scores[ok]
    gold_scores = gold_scores[ok]
    n = int(method_scores.size)
    if n == 0:
        return float("nan")

    k = max(1, min(int(np.ceil(float(k_frac) * n)), n))
    k10 = max(1, min(int(np.ceil(0.10 * n)), n))
    p = float(0.05 ** (1.0 / k10))
    method_order = np.argsort(method_scores)[::-1]
    gold_order = np.argsort(gold_scores)[::-1]

    method_seen: set[int] = set()
    gold_seen: set[int] = set()
    overlap = 0
    weighted = 0.0
    overlap_at_k = 0.0
    power = 1.0
    for depth in range(1, k + 1):
        m_idx = int(method_order[depth - 1])
        g_idx = int(gold_order[depth - 1])
        if m_idx == g_idx:
            overlap += 1
        else:
            if m_idx in gold_seen:
                overlap += 1
            if g_idx in method_seen:
                overlap += 1
        method_seen.add(m_idx)
        gold_seen.add(g_idx)
        agreement = overlap / float(depth)
        weighted += power * agreement
        overlap_at_k = agreement
        power *= p
    return float((1.0 - p) * weighted + power * overlap_at_k)


def _method_loaders(q: float) -> Dict[str, Callable[[Path, int], np.ndarray]]:
    return {
        "LiRA (M=8)": lambda exp, target: _load_attack(exp, target, "lira_refs", 8),
        "RMIA (M=8)": lambda exp, target: _load_attack(exp, target, "rmia_refs", 8),
        "Attack-R (M=16)": lambda exp, target: _load_attack(exp, target, "attackr_refs", 16),
        "LT-MLP (M=1)": _load_lt_mlp,
        "LT-IQR (M=0)": _load_lt_iqr,
        "LOSS (M=0)": _load_loss,
        "RARE (M=1)": lambda exp, target: load_rare_locked_scores(exp, target, num_refs=1, q=q),
    }


def collect_rows(
    benchmarks: Dict[str, Path],
    targets: Iterable[int],
    *,
    q: float,
) -> pd.DataFrame:
    loaders = _method_loaders(q)
    rows: List[dict] = []
    for data_name, exp in benchmarks.items():
        for target in targets:
            gold = _load_attack(exp, int(target), "lira_refs", 256)
            members = _load_keep_mask(exp, int(target), int(gold.shape[0]))
            gold_m = gold[members]
            for method in METHODS:
                scores = loaders[method](exp, int(target))
                scores_m = scores[members]
                for k_frac in K_FRACS:
                    rows.append(
                        {
                            "data": data_name,
                            "target": int(target),
                            "method": method,
                            "k": f"{int(k_frac * 100)}%",
                            "k_frac": float(k_frac),
                            "rbo": _rbo_ext_at_k(scores_m, gold_m, k_frac),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["data", "method", "k", "k_frac"], sort=False)["rbo"]
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
        "LiRA (M=8)": dict(color="#0072B2", marker="o", linestyle="-"),
        "RMIA (M=8)": dict(color="#E69F00", marker="s", linestyle="-"),
        "Attack-R (M=16)": dict(color="#009E73", marker="D", linestyle="-"),
        "LT-MLP (M=1)": dict(color="#CC79A7", marker="^", linestyle="-"),
        "LT-IQR (M=0)": dict(color="#56B4E9", marker="v", linestyle="--"),
        "LOSS (M=0)": dict(color="#8A8A8A", marker="x", linestyle=":"),
        "RARE (M=1)": dict(color="#D55E00", marker="*", linestyle="-"),
    }

    datasets = ["CIFAR10", "CIFAR100", "CINIC10"]
    x = np.arange(len(K_FRACS))
    xticklabels = [f"{int(k * 100)}%" for k in K_FRACS]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), sharey=True)
    for ax, data_name in zip(axes, datasets):
        subset = summary[summary["data"] == data_name]
        for method in METHODS:
            row = subset[subset["method"] == method].sort_values("k_frac")
            ax.plot(
                x,
                row["mean"].to_numpy(dtype=float),
                linewidth=1.3,
                markersize=3.8 if method != "RARE (M=1)" else 5.2,
                label=method,
                **style[method],
            )
        ax.set_title(data_name, fontweight="bold", pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(r"Top-$k$ (%)")
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.45)
        ax.grid(True, axis="x", color="#F0F0F0", linewidth=0.35)
        ax.set_ylim(-0.02, 0.70)
    axes[0].set_ylabel("RBO")

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


def write_tex(output_pdf: Path, tex_path: Path) -> None:
    include_path = output_pdf.name if output_pdf.parent == tex_path.parent else output_pdf
    include_s = str(include_path).replace("\\", "/")
    tex = rf"""\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{{include_s}}}
\caption{{Rank-biased overlap (RBO) ranking agreement with the LiRA-256 top-$k$\% prefix.}}
\label{{fig:ranking-rbo}}
\end{{figure*}}
"""
    tex_path.write_text(tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--q", type=float, default=0.50, help="LT-BTM-R bottom-q window for RARE")
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(DEFAULT_BENCHMARKS, args.targets, q=float(args.q))
    summary = summarize_rows(rows)
    output = Path(args.output)
    rows.to_csv(output.with_suffix(".per_target.csv"), index=False)
    summary.to_csv(output.with_suffix(".summary.csv"), index=False)
    plot_summary(summary, output)
    write_tex(output, output.with_suffix(".tex"))

    for data_name in ["CIFAR10", "CIFAR100", "CINIC10"]:
        print(data_name)
        wide = (
            summary[summary["data"] == data_name]
            .pivot(index="method", columns="k", values="mean")
            .loc[METHODS]
        )
        print(wide.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
