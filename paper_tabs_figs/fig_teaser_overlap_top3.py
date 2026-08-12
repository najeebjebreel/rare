"""Generate the teaser top-3% LiRA-overlap figure."""

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

from rare.analysis.utils import _budget_at_frac, score_avg_min_q_reference_losses


DEFAULT_EXPERIMENT = ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140"
DEFAULT_TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
DEFAULT_OUTPUT = ROOT / "paper_tabs_figs/figures/fig_teaser_overlap_top3.pdf"
USENIX_COLUMN_WIDTH_IN = (7.00 - 0.33) / 2.0
METHODS = [
    "LiRA (M=8)",
    "RMIA (M=8)",
    "Attack-R (M=16)",
    "LT-MLP (M=1)",
    "LT-IQR",
    "LT-BTM-R",
    "RARE (LT-IQR + LT-BTM-R)",
]


def _zscore_on_mask(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(scores.shape, np.nan, dtype=np.float32)
    valid = mask & np.isfinite(scores)
    vals = scores[valid].astype(np.float32, copy=False)
    if vals.size == 0:
        return out
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    out[valid] = vals - mu if sd <= 0 or not np.isfinite(sd) else (vals - mu) / sd
    return out


def _load_lira_gold(exp: Path, target: int, fpr: float) -> tuple[np.ndarray, np.ndarray]:
    attack_dir = exp / f"model_{int(target)}" / "attack_results"
    scores = np.load(attack_dir / "lira_refs_256_scores.npy").astype(np.float32)
    labels = np.load(attack_dir / "lira_refs_256_labels.npy").astype(bool)
    thresholds = np.load(
        attack_dir / "lira_refs_256_thresholds_by_fpr.npy",
        allow_pickle=True,
    ).item()
    if fpr not in thresholds:
        available = ", ".join(str(k) for k in sorted(thresholds))
        raise KeyError(f"FPR={fpr} missing for target {target}; available: {available}")
    vulnerable = (scores >= float(thresholds[fpr])) & labels
    return labels, vulnerable


def _load_attack_scores(exp: Path, target: int, attack: str, num_refs: int) -> np.ndarray:
    prefix = {"LiRA": "lira_refs", "RMIA": "rmia_refs", "Attack-R": "attackr_refs"}[attack]
    return np.load(
        exp / f"model_{int(target)}" / "attack_results" / f"{prefix}_{int(num_refs)}_scores.npy"
    ).astype(np.float32)


def _load_lt_iqr(exp: Path, target: int) -> np.ndarray:
    return np.load(exp / f"model_{int(target)}" / "lt_iqr_scores.npy").astype(np.float32)


def _load_lt_btm_r(exp: Path, target: int, num_refs: int, q: float) -> np.ndarray:
    target_dir = exp / f"model_{int(target)}"
    refs: List[np.ndarray] = []
    for ref_idx in range(int(num_refs)):
        matches = sorted(target_dir.glob(f"ref_losses_E*_*_ref{ref_idx}.npy"))
        if not matches:
            raise FileNotFoundError(f"No ref_losses_E*_*_ref{ref_idx}.npy in {target_dir}")
        refs.append(np.load(matches[0]).astype(np.float32))
    return score_avg_min_q_reference_losses(refs, q=q)


def _method_scores(exp: Path, target: int, members: np.ndarray, num_refs: int, q: float) -> Dict[str, np.ndarray]:
    lt_iqr = _load_lt_iqr(exp, target)
    lt_btm_r = _load_lt_btm_r(exp, target, num_refs=num_refs, q=q)
    rare_combo = _zscore_on_mask(lt_iqr, members) + _zscore_on_mask(lt_btm_r, members)
    return {
        "LiRA (M=8)": _load_attack_scores(exp, target, "LiRA", 8),
        "RMIA (M=8)": _load_attack_scores(exp, target, "RMIA", 8),
        "Attack-R (M=16)": _load_attack_scores(exp, target, "Attack-R", 16),
        "LT-MLP (M=1)": np.load(
            exp / f"model_{int(target)}" / "liu_mlp_scores_R1.npy"
        ).astype(np.float32),
        "LT-IQR": lt_iqr,
        "LT-BTM-R": lt_btm_r,
        "RARE (LT-IQR + LT-BTM-R)": rare_combo,
    }


def _top_indices(scores: np.ndarray, members: np.ndarray, top_frac: float) -> np.ndarray:
    """Top-k member indices, k = ceil(top_frac * n_valid_members).

    Uses the shared budget helper so the retrieval depth matches the
    Precision@k convention in rare.analysis.utils (no |V| truncation here:
    this figure counts overlap at a fixed budget, it is not a capped metric).
    """
    valid_members = np.where(members & np.isfinite(scores))[0]
    k = _budget_at_frac(float(top_frac), len(valid_members), "ceil")
    if k == 0:
        return np.asarray([], dtype=np.int64)
    return valid_members[np.argsort(scores[valid_members])[::-1][:k]]


def collect_rows(
    exp: Path,
    targets: Iterable[int],
    *,
    fpr: float,
    top_frac: float,
    num_refs: int,
    btm_q: float,
) -> pd.DataFrame:
    rows = []
    for target in targets:
        members, vulnerable = _load_lira_gold(exp, int(target), fpr)
        for method, scores in _method_scores(exp, int(target), members, num_refs, btm_q).items():
            top = _top_indices(scores, members, top_frac)
            overlap = int(vulnerable[top].sum())
            selected = int(top.size)
            rows.append(
                {
                    "target": int(target),
                    "method": method,
                    "selected": selected,
                    "lira_overlap": overlap,
                    "remaining": selected - overlap,
                    "precision": float(overlap / selected) if selected else float("nan"),
                    "gold_vulnerable": int(vulnerable.sum()),
                    "members": int(members.sum()),
                }
            )
    return pd.DataFrame(rows)


def _format_complementarity_counts(counts: Dict[float, float]) -> str:
    return ", ".join(f"{counts[k]:.0f}" for k in sorted(counts))


def _write_tex_snippet(
    output_pdf: Path,
    tex_path: Path,
    top_frac: float,
    complementarity_counts: Dict[float, float],
) -> None:
    include_path = output_pdf.name if output_pdf.parent == tex_path.parent else output_pdf
    include_s = str(include_path).replace("\\", "/")
    top_pct = f"{top_frac * 100:g}"
    label = output_pdf.stem
    if label.startswith("fig_"):
        label = label[4:]
    tex = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{{include_s}}}
\caption{{
Mean top-${top_pct}\%$ set per target on CIFAR-10 / WRN-28-2. Coloured:
samples also detected by LiRA-256 at FPR$=10^{{-4}}$; grey: remaining.
With a single reference model, \rare{{}} recovers more of the
high-budget vulnerable set than the target-only LT-IQR and the
same-budget supervised LT-MLP, at a fraction of the reference attacks'
shadow cost.
}}
\label{{fig:{label}}}
\end{{figure}}
"""
    tex_path.write_text(tex, encoding="utf-8")


def compute_complementarity_counts(
    exp: Path,
    targets: Iterable[int],
    *,
    fpr: float,
    top_fracs: Iterable[float],
    num_refs: int,
    btm_q: float,
) -> Dict[float, float]:
    counts: Dict[float, float] = {}
    for top_frac in top_fracs:
        per_target = []
        for target in targets:
            members, vulnerable = _load_lira_gold(exp, int(target), fpr)
            scores = _method_scores(exp, int(target), members, num_refs, btm_q)
            rare_top = set(_top_indices(scores["RARE (LT-IQR + LT-BTM-R)"], members, top_frac).tolist())
            ltiqr_top = set(_top_indices(scores["LT-IQR"], members, top_frac).tolist())
            extra = [idx for idx in rare_top - ltiqr_top if vulnerable[idx]]
            per_target.append(len(extra))
        counts[float(top_frac)] = float(np.mean(per_target))
    return counts


def plot_summary(rows: pd.DataFrame, output: Path, top_frac: float) -> None:
    grouped = rows.groupby("method", sort=False).agg(
        selected=("selected", "mean"),
        lira_overlap=("lira_overlap", "mean"),
        remaining=("remaining", "mean"),
    ).loc[METHODS]
    labels = [
        "LiRA (M=8)",
        "RMIA (M=8)",
        "Attack-R (M=16)",
        "LT-MLP (M=1)",
        "LT-IQR (M=0)",
        "LT-BTM-R (M=1)",
        "RARE (M=1)",
    ]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#F0E442", "#D55E00"]
    y = np.arange(len(METHODS))
    overlap = grouped["lira_overlap"].to_numpy(dtype=float)
    remaining = grouped["remaining"].to_numpy(dtype=float)
    x_max = max(750.0, float(np.ceil((overlap + remaining).max() / 50.0) * 50.0))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(USENIX_COLUMN_WIDTH_IN, 1.58), constrained_layout=True)
    ax.barh(y, overlap, height=0.62, color=colors, edgecolor="none")
    ax.barh(
        y,
        remaining,
        height=0.62,
        left=overlap,
        color="#E6E6E6",
        edgecolor="#D0D0D0",
        linewidth=0.35,
    )

    for i, hit in enumerate(overlap):
        # Dark text on light bars (e.g. the yellow LT-BTM-R), white otherwise.
        rgb = tuple(int(colors[i][j:j + 2], 16) / 255 for j in (1, 3, 5))
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        ax.text(
            hit * 0.52,
            i,
            f"Detected {hit:.0f}",
            ha="center",
            va="center",
            fontsize=5.8,
            color="white" if luminance < 0.6 else "#333333",
            fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_xlim(0, x_max)
    ax.set_xticks([0, 250, 500, 750] if x_max <= 750 else np.arange(0, x_max + 1, 250))
    ax.tick_params(axis="x", labelsize=7, length=2.0, width=0.45)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#DADADA", linestyle=":", linewidth=0.45)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["bottom"].set_linewidth(0.6)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--targets", type=int, nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-frac", type=float, default=0.03)
    parser.add_argument("--fpr", type=float, default=1e-4)
    parser.add_argument("--num-refs", type=int, default=1)
    parser.add_argument("--btm-q", type=float, default=0.50)
    args = parser.parse_args()

    exp = args.experiment_dir.resolve()
    output = args.output.resolve()
    rows = collect_rows(
        exp,
        args.targets,
        fpr=float(args.fpr),
        top_frac=float(args.top_frac),
        num_refs=int(args.num_refs),
        btm_q=float(args.btm_q),
    )
    summary = rows.groupby("method", sort=False).agg(
        selected_mean=("selected", "mean"),
        selected_std=("selected", "std"),
        lira_overlap_mean=("lira_overlap", "mean"),
        lira_overlap_std=("lira_overlap", "std"),
        remaining_mean=("remaining", "mean"),
        remaining_std=("remaining", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
    ).loc[METHODS]

    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output.with_suffix(".per_target.csv"), index=False)
    summary.to_csv(output.with_suffix(".summary.csv"))
    plot_summary(rows, output, top_frac=float(args.top_frac))
    complementarity_counts = compute_complementarity_counts(
        exp,
        args.targets,
        fpr=float(args.fpr),
        top_fracs=[0.01, 0.05, 0.10],
        num_refs=int(args.num_refs),
        btm_q=float(args.btm_q),
    )
    _write_tex_snippet(
        output,
        output.with_suffix(".tex"),
        top_frac=float(args.top_frac),
        complementarity_counts=complementarity_counts,
    )

    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.summary.csv')}")
    print(f"wrote {output.with_suffix('.tex')}")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("extra LiRA-vulnerable in RARE top-k but not LT-IQR top-k:")
    for frac, count in complementarity_counts.items():
        print(f"  top-{frac * 100:g}%: {count:.1f}")


if __name__ == "__main__":
    main()
