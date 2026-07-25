"""
Aggregate per-method runtime totals from `<exp>/runtime.csv`.

Each method's "Runtime (s)" per target is the sum of the rows it depends on:

  baseline_target = target_retrain_baseline                        (one per target)
  lt_iqr  = target_retrain_loss_trace + lt_iqr_score
  liu_mlp = target_retrain_loss_trace + liu_mlp_score              (per scored target)
  rare    = target_retrain_rare_trace + ref_retrain (sum over R refs)

Tracing-overhead estimates are derived by subtraction:

  loss_tracing_overhead = mean(target_retrain_loss_trace) - mean(target_retrain_baseline)
  rare_tracing_overhead = mean(target_retrain_rare_trace) - mean(target_retrain_baseline)

Outputs (next to runtime.csv):
  runtime_per_target.csv      one row per target, columns = method totals
  runtime_per_method.csv      per-method mean ± std across targets
  runtime_overhead.csv        baseline mean + tracing overhead estimates

Usage:
  python scripts/aggregate_runtimes.py --experiment-dir experiments/<bench>/<run>
"""

import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Phase → method-component mapping. reference retrain rows are summed per target across
# all refs; everything else is one row per target by construction.
_BASELINE_PHASE = "target_retrain_baseline"
_LOSS_TRACE_PHASE = "target_retrain_loss_trace"
_RARE_TRACE_PHASE = "target_retrain_rare_trace"
_LT_IQR_SCORE = "lt_iqr_score"
_LIU_MLP_SCORE = "liu_mlp_score"
_REF_RETRAIN = "ref_retrain"


def _per_target_seconds(df: pd.DataFrame, phase: str) -> pd.Series:
    """Sum `seconds` per `target_<t>` for rows of the given phase."""
    sub = df[df["phase"] == phase].copy()
    if sub.empty:
        return pd.Series(dtype="float64")
    sub["target"] = sub["item"].str.extract(r"target_(\d+)").astype("Int64")
    sub = sub.dropna(subset=["target"])
    return sub.groupby("target")["seconds"].sum().sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-dir", required=True, type=str)
    ap.add_argument(
        "--runtime-csv", type=str, default=None,
        help="Override path to runtime.csv (default: <experiment-dir>/runtime.csv).",
    )
    ap.add_argument(
        "--out-dir", type=str, default=None,
        help="Override directory for the three output CSVs (default: <experiment-dir>).",
    )
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    runtime_csv = Path(args.runtime_csv) if args.runtime_csv else (exp_dir / "runtime.csv")
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir

    if not runtime_csv.exists():
        print(f"runtime.csv not found at {runtime_csv}", file=sys.stderr)
        sys.exit(1)

    # The runtime.csv is appended-to by multiple scripts that pass different
    # `extra` keyword sets to append_runtime() (e.g. liu_mlp adds `R`).
    # Each write group keeps its own
    # header but later rows may have extra trailing columns, which pandas's C
    # parser rejects. Pull only the canonical 4 base columns by raw csv.
    import csv as _csv
    rows = []
    with open(runtime_csv, newline="") as f:
        reader = _csv.reader(f)
        header = None
        for row in reader:
            if not row:
                continue
            # The header row begins with "timestamp"; subsequent header re-writes
            # (if any) are skipped. Data rows always have timestamp as the
            # first field (an ISO-ish 19-char string).
            if header is None and row[0] == "timestamp":
                header = row
                continue
            if row[0] == "timestamp":
                continue
            if len(row) < 4:
                continue
            rows.append({"timestamp": row[0], "phase": row[1], "item": row[2], "seconds": row[3]})
    df = pd.DataFrame(rows)
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")

    baseline = _per_target_seconds(df, _BASELINE_PHASE)
    loss_trace = _per_target_seconds(df, _LOSS_TRACE_PHASE)
    rare_trace = _per_target_seconds(df, _RARE_TRACE_PHASE)
    lt_iqr_score = _per_target_seconds(df, _LT_IQR_SCORE)
    liu_score = _per_target_seconds(df, _LIU_MLP_SCORE)
    ref_retrain = _per_target_seconds(df, _REF_RETRAIN)

    # ----- Per-target totals -------------------------------------------------
    per_target = pd.DataFrame({
        "target_retrain_baseline_sec": baseline,
        "target_retrain_loss_trace_sec": loss_trace,
        "target_retrain_rare_trace_sec": rare_trace,
        "lt_iqr_score_sec": lt_iqr_score,
        "liu_mlp_score_sec": liu_score,
        "ref_retrain_total_sec": ref_retrain,
    })
    per_target["lt_iqr_total_sec"] = per_target["target_retrain_loss_trace_sec"].add(
        per_target["lt_iqr_score_sec"], fill_value=0.0,
    )
    per_target["liu_mlp_total_sec"] = per_target["target_retrain_loss_trace_sec"].add(
        per_target["liu_mlp_score_sec"], fill_value=0.0,
    )
    per_target["rare_total_sec"] = (
        per_target["target_retrain_rare_trace_sec"]
        .add(per_target["ref_retrain_total_sec"], fill_value=0.0)
    )
    per_target.index.name = "target_model"
    per_target = per_target.sort_index()

    per_target_path = out_dir / "runtime_per_target.csv"
    per_target.to_csv(per_target_path)
    print(f"Saved: {per_target_path}")

    # ----- Per-method mean ± std --------------------------------------------
    summary_cols = [
        "target_retrain_baseline_sec",
        "target_retrain_loss_trace_sec",
        "target_retrain_rare_trace_sec",
        "lt_iqr_score_sec",
        "liu_mlp_score_sec",
        "ref_retrain_total_sec",
        "lt_iqr_total_sec",
        "liu_mlp_total_sec",
        "rare_total_sec",
    ]
    rows = []
    for col in summary_cols:
        s = per_target[col].dropna()
        if s.empty:
            rows.append({"component": col, "n": 0, "mean_sec": np.nan, "std_sec": np.nan})
            continue
        rows.append({
            "component": col, "n": int(s.size),
            "mean_sec": float(s.mean()), "std_sec": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        })
    per_method = pd.DataFrame(rows)
    per_method_path = out_dir / "runtime_per_method.csv"
    per_method.to_csv(per_method_path, index=False)
    print(f"Saved: {per_method_path}")

    # ----- Overhead estimates -----------------------------------------------
    def _mean_or_nan(s: pd.Series) -> float:
        return float(s.dropna().mean()) if not s.dropna().empty else float("nan")

    bmean = _mean_or_nan(baseline)
    loss_mean = _mean_or_nan(loss_trace)
    rare_mean = _mean_or_nan(rare_trace)
    overhead_rows = [
        {"metric": "baseline_target_retrain_mean_sec", "value": bmean,
         "note": "no-tracing target retrain (one row per target)"},
        {"metric": "loss_trace_target_retrain_mean_sec", "value": loss_mean,
         "note": "loss-only tracing target retrain"},
        {"metric": "rare_trace_target_retrain_mean_sec", "value": rare_mean,
         "note": "(loss, margin) tracing target retrain"},
        {"metric": "loss_tracing_overhead_sec", "value": (loss_mean - bmean) if np.isfinite([loss_mean, bmean]).all() else float("nan"),
         "note": "loss_trace - baseline"},
        {"metric": "rare_tracing_overhead_sec", "value": (rare_mean - bmean) if np.isfinite([rare_mean, bmean]).all() else float("nan"),
         "note": "rare_trace - baseline"},
    ]
    overhead = pd.DataFrame(overhead_rows)
    overhead_path = out_dir / "runtime_overhead.csv"
    overhead.to_csv(overhead_path, index=False)
    print(f"Saved: {overhead_path}")

    print("\nMethod totals (mean ± std across targets):")
    for col in ("lt_iqr_total_sec", "liu_mlp_total_sec", "rare_total_sec"):
        s = per_target[col].dropna()
        if s.empty:
            print(f"  {col:<24} n=0")
        else:
            sd = float(s.std(ddof=1)) if s.size > 1 else 0.0
            print(f"  {col:<24} n={s.size:<3} mean={s.mean():8.2f}s  std={sd:7.2f}s")


if __name__ == "__main__":
    main()
