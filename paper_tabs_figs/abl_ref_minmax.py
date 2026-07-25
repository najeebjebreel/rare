"""Add REF-MIN and REF-MAX to the reference-statistic ablation.

Pure post-processing on the saved per-epoch reference loss traces (no model
training). Both new statistics belong to the same value-based family as
LT-BTM-R (mean of the bottom-q% epoch losses):

    REF-MIN_i = min_t l^R_{i,t}   -- the B_q=1 endpoint of that family
    REF-MEAN_i = mean_t l^R_{i,t} -- the B_q=E endpoint (already tabulated)
    REF-MAX_i = max_t l^R_{i,t}   -- the top endpoint (outside the family)

Sign convention matches every other reference-side row: higher reference loss
=> harder for the complement-trained reference => more vulnerable, so the raw
statistic is ranked with larger = more member-like (no negation).

This script reuses the ``tab_abl_final_vs_traj`` data-loading, LiRA-gold,
capped-P@k, z-score, and aggregation code paths verbatim (so the traces are the
same arrays feeding the existing rows), then writes only the four new-row
artifacts under ``results/``. It never rewrites the main table, so no existing
row can change.

Usage:
  python paper_tabs_figs/abl_ref_minmax.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "paper_tabs_figs")):
    if p not in sys.path:
        sys.path.insert(0, p)

import tab_abl_final_vs_traj as base  # noqa: E402
from rare.analysis.utils import score_avg_min_q_reference_losses  # noqa: E402

BTM_LABEL = "LT-BTM-R ($q{=}50\\%$)"
RARE_LABEL = "\\rare{} (LT-IQR + LT-BTM-R)"
REF_MIN = "REF-MIN"
REF_MAX = "REF-MAX"
REF_MIN_COMBINED = "LT-IQR + REF-MIN"
REF_MAX_COMBINED = "LT-IQR + REF-MAX"

# REF-MIN / REF-MAX placed adjacent to REF-MEAN in both blocks.
ROW_ORDER = [
    "LT-IQR",
    "REF-FINAL",
    "CALIB-FINAL",
    "REF-MEAN",
    REF_MIN,
    REF_MAX,
    BTM_LABEL,
    "LT-IQR + REF-FINAL",
    "LT-IQR + CALIB-FINAL",
    "LT-IQR + REF-MEAN",
    REF_MIN_COMBINED,
    REF_MAX_COMBINED,
    RARE_LABEL,
]

# New rows we actually emit (four), in table order.
NEW_ROWS = [REF_MIN, REF_MAX, REF_MIN_COMBINED, REF_MAX_COMBINED]

# Existing rows that must remain bit-identical (check 2). REF-FINAL / CALIB /
# REF-MEAN / LT-BTM-R / their combined forms live in the main CSV; REF-LAST
# lives in the companion abl_ref_last.csv.
MAIN_REGRESSION_ROWS = [
    "REF-FINAL",
    "CALIB-FINAL",
    "REF-MEAN",
    BTM_LABEL,
    "LT-IQR + REF-FINAL",
    "LT-IQR + CALIB-FINAL",
    "LT-IQR + REF-MEAN",
    RARE_LABEL,
]


def _display_label(label: str) -> str:
    return label.replace("\\rare{}", "RARE").replace("\\%", "%")


def _score_ref_min(ref: np.ndarray) -> np.ndarray:
    if ref.ndim != 2:
        raise ValueError(f"REF-MIN expects a (N, E) trace array, got {ref.shape}")
    return ref.min(axis=1).astype(np.float32)


def _score_ref_max(ref: np.ndarray) -> np.ndarray:
    if ref.ndim != 2:
        raise ValueError(f"REF-MAX expects a (N, E) trace array, got {ref.shape}")
    return ref.max(axis=1).astype(np.float32)


def _stats(ref: np.ndarray, tgt: np.ndarray) -> dict[str, np.ndarray]:
    stats = base._stats(ref, tgt)
    stats[REF_MIN] = _score_ref_min(ref)
    stats[REF_MAX] = _score_ref_max(ref)
    return stats


def _row_score(label: str, stats: dict[str, np.ndarray], members: np.ndarray) -> np.ndarray:
    if label == REF_MIN:
        return base._masked(stats[REF_MIN], members)
    if label == REF_MAX:
        return base._masked(stats[REF_MAX], members)
    if label == REF_MIN_COMBINED:
        return (
            base._zscore_ignore_nan(stats[REF_MIN], members)
            + base._zscore_ignore_nan(stats["LT-IQR"], members)
        )
    if label == REF_MAX_COMBINED:
        return (
            base._zscore_ignore_nan(stats[REF_MAX], members)
            + base._zscore_ignore_nan(stats["LT-IQR"], members)
        )
    return base._row_score(label, stats, members)


def evaluate() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list]]:
    """Return (results, check1_records). results[ds][label] = (T, K) precision."""
    results = {ds: {lab: [] for lab in ROW_ORDER} for ds in base.BENCHMARKS}
    # check 1: REF-MIN score vector == score_avg_min_q at B_q=1, per target.
    check1 = {"max_abs_diff": 0.0, "worst": None, "n": 0}
    for ds, exp in base.BENCHMARKS.items():
        for t in base.TARGETS:
            ref, tgt, members = base._load_traj(exp, t)
            stats = _stats(ref, tgt)

            # --- check 1: standalone REF-MIN vs LT-BTM-R at B_q=1 (k=1) ---
            E = int(ref.shape[1])
            q1 = 1.0 / E  # round(q1*E)=1 -> selects a single (minimum) epoch
            btm_bq1 = score_avg_min_q_reference_losses([ref], q=q1)
            d = float(np.max(np.abs(stats[REF_MIN].astype(np.float64) - btm_bq1.astype(np.float64))))
            check1["n"] += 1
            if d > check1["max_abs_diff"]:
                check1["max_abs_diff"] = d
                check1["worst"] = (ds, t)

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
    return results, check1


def summarize_cells(results):
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


def _load_main_cells(path: Path):
    """(method, dataset, k) -> (mean, std) from the main long CSV."""
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[(row["method"], row["dataset"], row["k"])] = (
                float(row["mean"]),
                float(row["std"]),
            )
    return out


def check_regression(cell_mean, cell_std) -> list[str]:
    """Existing rows must match the stored main table (bit-identical to 4 dp)."""
    stored = _load_main_cells(ROOT / "paper_tabs_figs/figures/tab_abl_final_vs_trajectory.csv")
    blocks = list(base.BENCHMARKS) + ["Avg"]
    errors = []
    for lab in MAIN_REGRESSION_ROWS:
        method = _display_label(lab)
        for blk, ds in enumerate(blocks):
            for j, k in enumerate(base.K_LABELS):
                key = (method, ds, k)
                if key not in stored:
                    errors.append(f"missing stored cell {key}")
                    continue
                exp_mean, exp_std = stored[key]
                gm, gs = float(cell_mean[lab][blk, j]), float(cell_std[lab][blk, j])
                if abs(gm - exp_mean) > 5e-4 or abs(gs - exp_std) > 5e-4:
                    errors.append(
                        f"{method} {ds} {k}: got {gm:.4f}+/-{gs:.4f}, stored {exp_mean:.4f}+/-{exp_std:.4f}"
                    )
    return errors


def write_csv(path: Path, cell_mean, cell_std) -> None:
    blocks = list(base.BENCHMARKS) + ["Avg"]
    mode = {REF_MIN: "alone", REF_MAX: "alone",
            REF_MIN_COMBINED: "combined", REF_MAX_COMBINED: "combined"}
    stat = {REF_MIN: "REF-MIN", REF_MAX: "REF-MAX",
            REF_MIN_COMBINED: "REF-MIN", REF_MAX_COMBINED: "REF-MAX"}
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "statistic", "mode", "k", "mean", "std"])
        for lab in NEW_ROWS:
            for blk, ds in enumerate(blocks):
                for j, k in enumerate(base.K_LABELS):
                    w.writerow([ds, stat[lab], mode[lab], k,
                                f"{cell_mean[lab][blk, j]:.4f}", f"{cell_std[lab][blk, j]:.4f}"])


def write_rows_tex(path: Path, cell_mean, cell_std, best) -> None:
    n_ds, nk = len(base.BENCHMARKS), len(base.K_FRACS)
    rows = [base._format_row(lab, cell_mean, cell_std, best, n_ds, nk) for lab in NEW_ROWS]
    header = ("% Drop-in rows for tab:abl-final-vs-traj (generated by abl_ref_minmax.py).\n"
              "% Standalone REF-MIN/REF-MAX go right after the REF-MEAN row;\n"
              "% combined LT-IQR + REF-MIN/REF-MAX right after the LT-IQR + REF-MEAN row.\n")
    path.write_text(header + "\n".join(rows) + "\n")


def markdown_table(cell_mean, cell_std) -> str:
    blocks = list(base.BENCHMARKS) + ["Avg"]
    lines = ["| Row | Dataset | P@1% | P@3% | P@5% | P@10% |",
             "|---|---|---|---|---|---|"]
    for lab in NEW_ROWS:
        for blk, ds in enumerate(blocks):
            cells = " | ".join(f"{cell_mean[lab][blk, j]:.2f}±{cell_std[lab][blk, j]:.2f}"
                               for j in range(len(base.K_LABELS)))
            lines.append(f"| {_display_label(lab)} | {ds} | {cells} |")
    return "\n".join(lines)


def check_ref_max_beats_ref_mean(cell_mean) -> list[str]:
    """Flag every cell where REF-MAX >= REF-MEAN (contradicts lower-tail rationale)."""
    blocks = list(base.BENCHMARKS) + ["Avg"]
    flags = []
    for (mx, mn, tag) in [(REF_MAX, "REF-MEAN", "alone"),
                          (REF_MAX_COMBINED, "LT-IQR + REF-MEAN", "combined")]:
        for blk, ds in enumerate(blocks):
            for j, k in enumerate(base.K_LABELS):
                a, b = float(cell_mean[mx][blk, j]), float(cell_mean[mn][blk, j])
                if a >= b:
                    flags.append(f"[{tag}] {ds}/{k}: {_display_label(mx)}={a:.4f} >= "
                                 f"{_display_label(mn)}={b:.4f}")
    return flags


def diagnostics() -> str:
    """CIFAR10-member diagnostics for the paper text (checks 4 and 5)."""
    exp = base.BENCHMARKS["CIFAR10"]
    argmax_all, argmin_all = [], []
    refmax_all, reffinal_all, refmin_all, btm50_all = [], [], [], []
    E = None
    for t in base.TARGETS:
        ref, tgt, members = base._load_traj(exp, t)
        E = int(ref.shape[1])
        rm = ref[members]  # (n_mem, E)
        argmax_all.append(np.argmax(rm, axis=1))
        argmin_all.append(np.argmin(rm, axis=1))
        refmax_all.append(rm.max(axis=1))
        refmin_all.append(rm.min(axis=1))
        reffinal_all.append(rm[:, -1])
        btm50_all.append(score_avg_min_q_reference_losses([ref], q=0.50)[members])
    argmax = np.concatenate(argmax_all)
    argmin = np.concatenate(argmin_all)
    refmax = np.concatenate(refmax_all)
    refmin = np.concatenate(refmin_all)
    reffinal = np.concatenate(reffinal_all)
    btm50 = np.concatenate(btm50_all)

    final25_start = int(np.ceil(0.75 * E))  # epoch index into final 25% of training
    frac_max_final25 = float(np.mean(argmax >= final25_start))
    frac_min_final25 = float(np.mean(argmin >= final25_start))
    rho_max_final = float(spearmanr(refmax, reffinal).statistic)
    rho_min_btm = float(spearmanr(refmin, btm50).statistic)

    lines = [
        f"CIFAR10 target-member diagnostics (n_members={argmax.size}, E={E}):",
        f"  argmax_t epoch: median={int(np.median(argmax))}  "
        f"mean={argmax.mean():.1f}  p25={int(np.percentile(argmax,25))} "
        f"p75={int(np.percentile(argmax,75))}",
        f"  argmin_t epoch: median={int(np.median(argmin))}  "
        f"mean={argmin.mean():.1f}  p25={int(np.percentile(argmin,25))} "
        f"p75={int(np.percentile(argmin,75))}",
        f"  fraction with max in final 25% of training (epoch >= {final25_start}): "
        f"{frac_max_final25:.4f}",
        f"  (fraction with min in final 25%: {frac_min_final25:.4f})",
        f"  Spearman(REF-MAX, REF-FINAL) over members = {rho_max_final:.4f}",
        f"  Spearman(REF-MIN, LT-BTM-R q=0.5) over members = {rho_min_btm:.4f}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    results, check1 = evaluate()
    cell_mean, cell_std, best = summarize_cells(results)

    reg_errors = check_regression(cell_mean, cell_std)
    c1_tol = 1e-6
    c1_pass = check1["max_abs_diff"] <= c1_tol

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if reg_errors:
        print("CHECK 2 (existing rows bit-identical): FAIL")
        for e in reg_errors[:40]:
            print("  " + e)
        raise SystemExit("Existing-row regression failed; not writing REF-MIN/REF-MAX outputs.")

    write_csv(args.out_dir / "abl_ref_minmax.csv", cell_mean, cell_std)
    write_rows_tex(args.out_dir / "abl_ref_minmax_rows.tex", cell_mean, cell_std, best)

    print("=== NEW ROWS (markdown) ===")
    print(markdown_table(cell_mean, cell_std))

    print("\n=== CONSISTENCY CHECKS ===")
    worst = check1["worst"]
    print(f"Check 1 REF-MIN == LT-BTM-R@B_q=1: "
          f"{'PASS' if c1_pass else 'FAIL'} "
          f"(max|diff|={check1['max_abs_diff']:.3e} over {check1['n']} targets, "
          f"tol={c1_tol:.0e}, worst={worst})")
    print(f"Check 2 existing rows bit-identical to stored table: PASS "
          f"({len(MAIN_REGRESSION_ROWS)} rows x 4 blocks x 4 k within 5e-4)")
    print("Check 3 same trace arrays as existing rows: PASS by construction "
          "(base._load_traj / base._stats reused; no checkpoint re-derivation)")

    print("\n=== REF-MAX vs REF-MEAN flags ===")
    mm_flags = check_ref_max_beats_ref_mean(cell_mean)
    if mm_flags:
        print("PROMINENT FLAG: REF-MAX reaches/exceeds REF-MEAN (contradicts lower-tail rationale):")
        for fl in mm_flags:
            print("  " + fl)
    else:
        print("REF-MAX stays strictly below REF-MEAN in every cell (both blocks).")

    print("\n=== DIAGNOSTICS (paper text) ===")
    diag = diagnostics()
    print(diag)

    (args.out_dir / "abl_ref_minmax_summary.txt").write_text(
        markdown_table(cell_mean, cell_std) + "\n\n"
        + f"Check1 max|diff|={check1['max_abs_diff']:.3e} pass={c1_pass}\n"
        + ("REF-MAX>=REF-MEAN flags:\n" + "\n".join(mm_flags) if mm_flags
           else "REF-MAX < REF-MEAN in every cell.\n")
        + "\n\n" + diag + "\n"
    )
    print("\nWrote:")
    print(f"  {args.out_dir / 'abl_ref_minmax.csv'}")
    print(f"  {args.out_dir / 'abl_ref_minmax_rows.tex'}")
    print(f"  {args.out_dir / 'abl_ref_minmax_summary.txt'}")


if __name__ == "__main__":
    main()
