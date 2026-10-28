"""Generate the qualitative four-example complementarity figure.

Picks four CIFAR-10 training images that are LiRA-256-vulnerable at the
given FPR and illustrate how the two RARE components complement each other:

  (a) both     : top-k% of LT-IQR, LT-BTM-R, and RARE
  (b) IQR-only : top-k% of LT-IQR and RARE, LT-BTM-R outside the exclusion
                 margin (default: outside top-10%)
  (c) BTM-only : top-k% of LT-BTM-R and RARE, LT-IQR outside the exclusion
                 margin
  (d) RARE-only: top-k% of RARE with BOTH components outside top-k%.
                 (No wider margin is feasible here: for the z-sum to reach
                 the top-k% while neither component does, both must sit just
                 below their own cutoff -- which is exactly the point.)

Each column shows the image with its per-epoch target and reference loss
trajectories underneath, the Q25-Q75 IQR band read by LT-IQR, and the
bottom-q reference epochs read by LT-BTM-R.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_tabs_figs._tex import figure_env
from rare.analysis.utils import score_avg_min_q_reference_losses

DEFAULT_EXPERIMENT = ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140"
DEFAULT_TARGET = 25
# Locked exemplars for the paper (cases a-d on target 25); pass --auto to
# re-run criterion-based selection instead. Panels (b)/(c) are the samples
# with the farthest missing-component rank subject to RARE staying top-3%.
DEFAULT_INDICES = [16002, 32909, 938, 6490]
DEFAULT_OUTPUT = ROOT / "paper_tabs_figs/figures/fig_qualitative_examples.pdf"
USENIX_TEXT_WIDTH_IN = 7.00 - 0.33

COLOR_TARGET = "#0072B2"
COLOR_REF = "#E69F00"
COLOR_BTM = "#D55E00"

CASES = [
    ("a", "Both components"),
    ("b", "LT-IQR only"),
    ("c", "LT-BTM-R only"),
    ("d", "RARE only"),
]


def _load_trajectories(exp: Path, target: int, ref_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    target_dir = exp / f"model_{int(target)}"
    tgt = np.load(target_dir / "per_epoch_losses.npy").astype(np.float32).T  # (N, E)
    matches = sorted(target_dir.glob(f"ref_losses_E*_*_ref{ref_idx}.npy"))
    if not matches:
        raise FileNotFoundError(f"No ref_losses_E*_*_ref{ref_idx}.npy in {target_dir}")
    ref = np.load(matches[0]).astype(np.float32)  # (N, E)
    if ref.shape != tgt.shape:
        raise ValueError(f"target {tgt.shape} vs reference {ref.shape} trajectory shape mismatch")
    return tgt, ref


def _load_lira_gold(exp: Path, target: int, fpr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return labels, vulnerable, scores


def _member_rank_frac(scores: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Fractional rank among members: 0 = highest score, ~1 = lowest."""
    out = np.full(scores.shape, np.nan, dtype=np.float32)
    idx = np.where(members & np.isfinite(scores))[0]
    order = np.argsort(scores[idx])[::-1]
    out[idx[order]] = np.arange(idx.size, dtype=np.float32) / float(idx.size)
    return out


def select_examples(
    vulnerable: np.ndarray,
    r_iqr: np.ndarray,
    r_btm: np.ndarray,
    r_rare: np.ndarray,
    *,
    top_frac: float,
    excl_frac: float,
) -> Dict[str, int]:
    """One exemplar index per case; furthest-from-the-boundary candidates win."""

    def pick(mask: np.ndarray, clarity: np.ndarray, case: str) -> int:
        cand = np.where(mask)[0]
        if cand.size == 0:
            raise RuntimeError(
                f"No candidate for case ({case}) with top_frac={top_frac}, "
                f"excl_frac={excl_frac}; try another target or a looser margin."
            )
        return int(cand[np.argmax(clarity[cand])])

    in_top = lambda r: r < top_frac  # noqa: E731
    chosen: Dict[str, int] = {}
    chosen["a"] = pick(
        vulnerable & in_top(r_iqr) & in_top(r_btm) & in_top(r_rare),
        -r_rare,  # best RARE rank
        "a",
    )
    chosen["b"] = pick(
        vulnerable & in_top(r_iqr) & in_top(r_rare) & (r_btm > excl_frac),
        r_btm,  # LT-BTM-R misses it as badly as possible
        "b",
    )
    chosen["c"] = pick(
        vulnerable & in_top(r_btm) & in_top(r_rare) & (r_iqr > excl_frac),
        r_iqr,  # LT-IQR misses it as badly as possible
        "c",
    )
    chosen["d"] = pick(
        vulnerable & in_top(r_rare) & (r_iqr > top_frac) & (r_btm > top_frac),
        np.minimum(r_iqr, r_btm),  # both components as far outside as possible
        "d",
    )
    return chosen


def _load_cifar10_images(indices: List[int]) -> tuple[List[np.ndarray], List[str]]:
    from PIL import Image
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=str(ROOT / "data"), train=True, download=False)
    images = [
        np.asarray(
            Image.fromarray(ds.data[i]).resize((1024, 1024), Image.LANCZOS)
        )
        for i in indices
    ]
    names = [ds.classes[int(ds.targets[i])] for i in indices]
    return images, names


def plot_examples(
    chosen: Dict[str, int],
    images: List[np.ndarray],
    class_names: List[str],
    tgt: np.ndarray,
    ref: np.ndarray,
    r_iqr: np.ndarray,
    r_btm: np.ndarray,
    r_rare: np.ndarray,
    output: Path,
    *,
    btm_q: float,
) -> None:
    E = tgt.shape[1]
    epochs = np.arange(1, E + 1)
    k_btm = max(1, int(round(btm_q * E)))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2,
        len(CASES),
        figsize=(USENIX_TEXT_WIDTH_IN, 3.55),
        gridspec_kw={"height_ratios": [1.05, 1.2]},
        constrained_layout=True,
    )
    fig.get_layout_engine().set(hspace=0.06, h_pad=0.02)
    y_max = max(
        max(float(tgt[chosen[key]].max()), float(ref[chosen[key]].max()))
        for key, _ in CASES
    )

    for col, ((key, title), image, cname) in enumerate(zip(CASES, images, class_names)):
        i = chosen[key]
        ax_img, ax = axes[0, col], axes[1, col]

        # interpolation="none": embed the pre-upscaled pixels untouched so the
        # PDF carries the full 1024px Lanczos render.
        ax_img.imshow(image, interpolation="none")
        ax_img.set_title(f"({key}) {title}", fontsize=7, pad=2.5)
        ax_img.text(
            0.5, -0.10, cname, transform=ax_img.transAxes,
            ha="center", va="top", fontsize=6, style="italic",
        )
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for side in ax_img.spines.values():
            side.set_linewidth(0.5)

        ax.plot(epochs, tgt[i], color=COLOR_TARGET, lw=0.7, label="target loss")

        # Reference trajectory with its bottom-q epochs and their mean (LT-BTM-R).
        ax.plot(epochs, ref[i], color=COLOR_REF, lw=0.7, label="reference loss")
        b_idx = np.argpartition(ref[i], k_btm - 1)[:k_btm]
        ax.scatter(
            epochs[b_idx], ref[i][b_idx], s=1.5, color=COLOR_BTM,
            zorder=3, label=f"bottom-{btm_q * 100:.0f}% ref. epochs",
        )
        btm_val = float(ref[i][b_idx].mean())
        ax.axhline(btm_val, color=COLOR_BTM, ls="--", lw=0.6, label="LT-BTM-R")

        ax.set_xlim(1, E)
        ax.set_ylim(0, y_max * 1.04)
        ax.set_xlabel("Epoch", fontsize=6.5, labelpad=1.5)
        if col == 0:
            ax.set_ylabel("Cross-entropy loss", fontsize=6.5)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=2.0, width=0.45)
        ax.set_axisbelow(True)
        ax.grid(True, color="#DADADA", linestyle=":", linewidth=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#333333")
            ax.spines[side].set_linewidth(0.6)

        ax.text(
            0.5, -0.28,
            f"IQR {r_iqr[i] * 100:.1f}% $\\cdot$ BTM-R {r_btm[i] * 100:.1f}%\n"
            f"RARE {r_rare[i] * 100:.1f}%",
            transform=ax.transAxes, ha="center", va="top", fontsize=6,
            linespacing=1.4,
        )

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.075),
        ncol=4, fontsize=6, frameon=False,
        handlelength=1.4, columnspacing=1.2,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


CAPTION = r"""
Four CIFAR10 members flagged as vulnerable by LiRA-256 at
FPR$=10^{-4}$ (WRN-28-2 target). Per-epoch target (blue) and
single-reference (orange) losses; dots mark the reference's
bottom-$50\%$ checkpoints, whose mean defines LT-BTM-R (dashed).
Lower percentages indicate more vulnerability
({\em e.g.} 1\% means among the top 1\% of vulnerability ranks). 
Panels show, in turn, a member recovered by both
components~(a), by target-side LT-IQR only~(b), by reference-side
LT-BTM-R only~(c), and by neither component alone but by their standardized
sum~(d).
"""


def _write_tex_snippet(output_pdf: Path, tex_path: Path) -> None:
    tex = figure_env(output_pdf.name, CAPTION, "fig:qualitative_examples",
                     width="0.75\\textwidth")
    tex_path.write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-frac", type=float, default=0.03)
    parser.add_argument("--excl-frac", type=float, default=0.10)
    parser.add_argument("--fpr", type=float, default=1e-4)
    parser.add_argument("--btm-q", type=float, default=0.50)
    parser.add_argument(
        "--indices", type=int, nargs=4, default=DEFAULT_INDICES,
        metavar=("A", "B", "C", "D"),
        help="The four sample indices (cases a-d); defaults to the locked paper exemplars.",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Ignore --indices and auto-select exemplars by the case criteria.",
    )
    args = parser.parse_args()

    exp = args.experiment_dir.resolve()
    output = args.output.resolve()
    target = int(args.target)

    tgt, ref = _load_trajectories(exp, target)
    members, vulnerable, lira_scores = _load_lira_gold(exp, target, float(args.fpr))

    lt_iqr = np.quantile(tgt, 0.75, axis=1) - np.quantile(tgt, 0.25, axis=1)
    lt_btm_r = score_avg_min_q_reference_losses([ref], q=float(args.btm_q))
    r_iqr = _member_rank_frac(lt_iqr, members)
    r_btm = _member_rank_frac(lt_btm_r, members)
    z = lambda s: (s - s[members].mean()) / s[members].std()  # noqa: E731
    r_rare = _member_rank_frac(z(lt_iqr) + z(lt_btm_r), members)

    if not args.auto:
        chosen = {key: int(idx) for (key, _), idx in zip(CASES, args.indices)}
    else:
        chosen = select_examples(
            vulnerable, r_iqr, r_btm, r_rare,
            top_frac=float(args.top_frac), excl_frac=float(args.excl_frac),
        )

    indices = [chosen[key] for key, _ in CASES]
    images, class_names = _load_cifar10_images(indices)
    plot_examples(
        chosen, images, class_names, tgt, ref, r_iqr, r_btm, r_rare, output,
        btm_q=float(args.btm_q),
    )
    _write_tex_snippet(output, output.with_suffix(".tex"))

    rows = []
    for (key, title), idx, cname in zip(CASES, indices, class_names):
        rows.append(
            {
                "case": key,
                "title": title,
                "sample_index": idx,
                "class": cname,
                "lira_score": float(lira_scores[idx]),
                "rank_lt_iqr": float(r_iqr[idx]),
                "rank_lt_btm_r": float(r_btm[idx]),
                "rank_rare": float(r_rare[idx]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output.with_suffix(".selected.csv"), index=False)

    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.tex')}")
    print(f"wrote {output.with_suffix('.selected.csv')}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
