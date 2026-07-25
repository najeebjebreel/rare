"""Add REF-LAST to the reference-statistic ablation without touching the table.

REF-LAST is the mean reference loss over the last half of the reference
trajectory, i.e. ``ref[:, E//2:].mean(axis=1)``. This script reuses the
``tab_abl_final_vs_traj`` data-loading, LiRA-gold, capped-P@k, z-score, and
aggregation code paths, then writes only the new REF-LAST artifacts under
``results/``.

Usage:
  python paper_tabs_figs/abl_ref_last.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "paper_tabs_figs")):
    if p not in sys.path:
        sys.path.insert(0, p)

import tab_abl_final_vs_traj as base  # noqa: E402

BTM_LABEL = "LT-BTM-R ($q{=}50\\%$)"
RARE_LABEL = "\\rare{} (LT-IQR + LT-BTM-R)"
REF_LAST = "REF-LAST"
REF_LAST_COMBINED = "LT-IQR + REF-LAST"

ROW_ORDER = [
    "LT-IQR",
    "REF-FINAL",
    "CALIB-FINAL",
    "REF-MEAN",
    REF_LAST,
    BTM_LABEL,
    "LT-IQR + REF-FINAL",
    "LT-IQR + CALIB-FINAL",
    "LT-IQR + REF-MEAN",
    REF_LAST_COMBINED,
    RARE_LABEL,
]

CSV_ROWS = [
    "REF-FINAL",
    "REF-MEAN",
    BTM_LABEL,
    REF_LAST,
    "LT-IQR + REF-FINAL",
    "LT-IQR + REF-MEAN",
    RARE_LABEL,
    REF_LAST_COMBINED,
]

REGRESSION_ROWS = [
    "REF-FINAL",
    "REF-MEAN",
    BTM_LABEL,
    "LT-IQR + REF-FINAL",
    "LT-IQR + REF-MEAN",
    RARE_LABEL,
]


def _display_label(label: str) -> str:
    return label.replace("\\rare{}", "RARE").replace("\\%", "%")


def _score_ref_last(ref: np.ndarray, *, seed: int) -> np.ndarray:
    if ref.ndim != 2:
        raise ValueError(f"REF-LAST expects a (N, E) trace array, got {ref.shape}")
    n, epochs = ref.shape
    start = epochs // 2
    window_count = epochs - start
    btm_count = int(math.ceil(0.5 * epochs))
    assert window_count == btm_count, (window_count, btm_count, epochs)
    assert window_count == 50, f"expected 50 REF-LAST checkpoints, got {window_count}"

    score = ref[:, start:].mean(axis=1).astype(np.float32)

    # Deterministic spot-check of a few random samples against the manual mean.
    rng = np.random.default_rng(seed)
    sample_count = min(5, n)
    idx = rng.choice(n, size=sample_count, replace=False)
    manual = ref[idx, start:].mean(axis=1).astype(np.float32)
    np.testing.assert_allclose(score[idx], manual, rtol=0.0, atol=1e-7)
    return score


def _stats(ref: np.ndarray, tgt: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    stats = base._stats(ref, tgt)
    stats[REF_LAST] = _score_ref_last(ref, seed=seed)
    return stats


def _row_score(label: str, stats: dict[str, np.ndarray], members: np.ndarray) -> np.ndarray:
    if label == REF_LAST:
        return base._masked(stats[REF_LAST], members)
    if label == REF_LAST_COMBINED:
        return (
            base._zscore_ignore_nan(stats[REF_LAST], members)
            + base._zscore_ignore_nan(stats["LT-IQR"], members)
        )
    return base._row_score(label, stats, members)


def evaluate() -> dict[str, dict[str, np.ndarray]]:
    results = {ds: {lab: [] for lab in ROW_ORDER} for ds in base.BENCHMARKS}
    for ds_index, (ds, exp) in enumerate(base.BENCHMARKS.items()):
        for t in base.TARGETS:
            ref, tgt, members = base._load_traj(exp, t)
            stats = _stats(ref, tgt, seed=1009 + 97 * ds_index + int(t))
            gold, mem_mask, vulnerable = base.mt._load_lira_gold(exp, t, base.FPR)
            assert np.array_equal(mem_mask, members)
            for lab in ROW_ORDER:
                pr = base.tail_metrics_at_k(
                    _row_score(lab, stats, members),
                    members,
                    base.K_FRACS,
                    vulnerable=vulnerable,
                    gold_scores=gold,
                )
                results[ds][lab].append(
                    [float(pr.iloc[i]["precision"]) for i in range(len(base.K_FRACS))]
                )
    for ds in results:
        for lab in results[ds]:
            results[ds][lab] = np.asarray(results[ds][lab], dtype=np.float64)
    return results


def summarize_cells(
    results: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[int, int], str]]:
    ds_names = list(base.BENCHMARKS)
    nk = len(base.K_FRACS)
    cell_mean = {lab: np.zeros((len(ds_names) + 1, nk), dtype=np.float64) for lab in ROW_ORDER}
    cell_std = {lab: np.zeros((len(ds_names) + 1, nk), dtype=np.float64) for lab in ROW_ORDER}
    for lab in ROW_ORDER:
        ds_means = []
        for di, ds in enumerate(ds_names):
            arr = results[ds][lab]
            cell_mean[lab][di] = arr.mean(axis=0)
            cell_std[lab][di] = arr.std(axis=0, ddof=1)
            ds_means.append(arr.mean(axis=0))
        ds_means_arr = np.asarray(ds_means, dtype=np.float64)
        cell_mean[lab][-1] = ds_means_arr.mean(axis=0)
        cell_std[lab][-1] = ds_means_arr.std(axis=0, ddof=1)

    best = {}
    for blk in range(len(ds_names) + 1):
        for j in range(nk):
            best[(blk, j)] = max(ROW_ORDER, key=lambda lab: cell_mean[lab][blk, j])
    return cell_mean, cell_std, best


def _load_stored_cells(path: Path) -> dict[tuple[str, str, str], tuple[float, float]]:
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[(row["method"], row["dataset"], row["k"])] = (
                float(row["mean"]),
                float(row["std"]),
            )
    return out


def check_regression(cell_mean: dict[str, np.ndarray], cell_std: dict[str, np.ndarray]) -> None:
    stored_path = ROOT / "paper_tabs_figs/figures/tab_abl_final_vs_trajectory.csv"
    stored = _load_stored_cells(stored_path)
    blocks = list(base.BENCHMARKS) + ["Avg"]
    errors = []
    for lab in REGRESSION_ROWS:
        method = _display_label(lab)
        for blk, ds in enumerate(blocks):
            for j, k in enumerate(base.K_LABELS):
                key = (method, ds, k)
                if key not in stored:
                    errors.append(f"missing stored cell {key}")
                    continue
                exp_mean, exp_std = stored[key]
                got_mean = float(cell_mean[lab][blk, j])
                got_std = float(cell_std[lab][blk, j])
                if abs(got_mean - exp_mean) > 0.005 or abs(got_std - exp_std) > 0.005:
                    errors.append(
                        f"{method} {ds} {k}: got {got_mean:.4f}+/-{got_std:.4f}, "
                        f"stored {exp_mean:.4f}+/-{exp_std:.4f}"
                    )
    if errors:
        msg = "\n".join(errors[:40])
        extra = "" if len(errors) <= 40 else f"\n... {len(errors) - 40} more"
        raise SystemExit(
            "Regression check failed against stored tab_abl_final_vs_trajectory.csv; "
            "not writing REF-LAST outputs.\n" + msg + extra
        )


def _stat_and_mode(label: str) -> tuple[str, str]:
    if label.startswith("LT-IQR + REF-FINAL"):
        return "REF-FINAL", "combined"
    if label.startswith("LT-IQR + REF-MEAN"):
        return "REF-MEAN", "combined"
    if label == RARE_LABEL:
        return "LT-BTM-R", "combined"
    if label == REF_LAST_COMBINED:
        return REF_LAST, "combined"
    if label == BTM_LABEL:
        return "LT-BTM-R", "alone"
    return label, "alone"


def write_csv(path: Path, cell_mean: dict[str, np.ndarray], cell_std: dict[str, np.ndarray]) -> None:
    blocks = list(base.BENCHMARKS) + ["Avg"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "statistic", "mode", "k", "mean", "std"])
        for lab in CSV_ROWS:
            statistic, mode = _stat_and_mode(lab)
            for blk, ds in enumerate(blocks):
                for j, k in enumerate(base.K_LABELS):
                    writer.writerow(
                        [
                            ds,
                            statistic,
                            mode,
                            k,
                            f"{cell_mean[lab][blk, j]:.4f}",
                            f"{cell_std[lab][blk, j]:.4f}",
                        ]
                    )


def write_rows_tex(
    path: Path,
    cell_mean: dict[str, np.ndarray],
    cell_std: dict[str, np.ndarray],
    best: dict[tuple[int, int], str],
) -> None:
    rows = [
        base._format_row(REF_LAST, cell_mean, cell_std, best, len(base.BENCHMARKS), len(base.K_FRACS)),
        base._format_row(
            REF_LAST_COMBINED, cell_mean, cell_std, best, len(base.BENCHMARKS), len(base.K_FRACS)
        ),
    ]
    path.write_text("\n".join(rows) + "\n")


def _diff_line(
    cell_mean: dict[str, np.ndarray],
    lhs: str,
    rhs: str,
    blk: int,
) -> str:
    diffs = (cell_mean[lhs][blk] - cell_mean[rhs][blk]) * 100.0
    return ", ".join(f"{base.K_LABELS[j]} {diffs[j]:+.2f} pp" for j in range(len(base.K_LABELS)))


def build_summary(cell_mean: dict[str, np.ndarray]) -> str:
    blocks = list(base.BENCHMARKS) + ["Avg"]
    comparisons = [
        ("REF-FINAL", "LT-IQR + REF-FINAL"),
        ("REF-MEAN", "LT-IQR + REF-MEAN"),
        (BTM_LABEL, RARE_LABEL),
    ]
    lines = [
        "REF-LAST comparison summary",
        "Signed differences are REF-LAST minus the named baseline.",
        "",
        "Mode: alone",
    ]
    for blk, ds in enumerate(blocks):
        lines.append(f"{ds}:")
        for alone, _combined in comparisons:
            lines.append(f"  vs {_display_label(alone)}: {_diff_line(cell_mean, REF_LAST, alone, blk)}")

    lines.extend(["", "Mode: combined with LT-IQR"])
    for blk, ds in enumerate(blocks):
        lines.append(f"{ds}:")
        for alone, combined in comparisons:
            lines.append(
                f"  vs LT-IQR + {_display_label(alone)}: "
                f"{_diff_line(cell_mean, REF_LAST_COMBINED, combined, blk)}"
            )

    flags = []
    avg_blk = len(blocks) - 1
    for j, k in enumerate(base.K_LABELS):
        if cell_mean[REF_LAST][avg_blk, j] >= cell_mean[BTM_LABEL][avg_blk, j]:
            flags.append(
                f"REF-LAST >= LT-BTM-R on Avg/{k}: "
                f"{cell_mean[REF_LAST][avg_blk, j]:.4f} vs {cell_mean[BTM_LABEL][avg_blk, j]:.4f}"
            )
        if cell_mean[REF_LAST_COMBINED][avg_blk, j] >= cell_mean[RARE_LABEL][avg_blk, j]:
            flags.append(
                f"LT-IQR + REF-LAST >= RARE on Avg/{k}: "
                f"{cell_mean[REF_LAST_COMBINED][avg_blk, j]:.4f} vs {cell_mean[RARE_LABEL][avg_blk, j]:.4f}"
            )
    lines.append("")
    if flags:
        lines.append("PROMINENT FLAG: REF-LAST reaches or exceeds LT-BTM-R on the Avg column.")
        lines.extend(f"  {flag}" for flag in flags)
    else:
        lines.append("Ordering check: REF-LAST stays below LT-BTM-R on every Avg-column k.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    results = evaluate()
    cell_mean, cell_std, best = summarize_cells(results)
    check_regression(cell_mean, cell_std)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "abl_ref_last.csv", cell_mean, cell_std)
    write_rows_tex(args.out_dir / "abl_ref_last_rows.tex", cell_mean, cell_std, best)
    summary = build_summary(cell_mean)
    (args.out_dir / "abl_ref_last_summary.txt").write_text(summary)

    print(summary)
    print("Wrote:")
    print(f"  {args.out_dir / 'abl_ref_last.csv'}")
    print(f"  {args.out_dir / 'abl_ref_last_rows.tex'}")
    print(f"  {args.out_dir / 'abl_ref_last_summary.txt'}")


if __name__ == "__main__":
    main()
