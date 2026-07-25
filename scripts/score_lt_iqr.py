"""
Precompute LT-IQR vulnerability scores per target.

Loads each target's `per_epoch_losses.npy` (produced by
`scripts/train.py --trace-mode loss_only` or `full`),
computes the interquartile range of the per-sample loss trajectory
across the specified epoch window, and writes the result to
`<experiment_dir>/model_<t>/lt_iqr_scores.npy`.

LT-IQR (Pollock et al., USENIX'25) is computed only over member samples;
non-member rows are returned as NaN.

Usage
-----
  python scripts/score_lt_iqr.py --experiment-dir <exp>
  python scripts/score_lt_iqr.py --experiment-dir <exp> --targets 0 25 50
"""

import _bootstrap  # noqa: F401

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

from rare.analysis.utils import compute_lt_iqr_scores
from rare.utils.common import append_runtime, setup_logger


def _read_target_list(experiment_dir: str) -> List[int]:
    cfg_path = Path(experiment_dir) / "attack_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"attack_config.yaml not found at {cfg_path}; pass --targets explicitly"
        )
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return [int(t) for t in cfg.get("target_models", [])]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-dir", required=True, type=str)
    ap.add_argument("--targets", type=int, nargs="*", default=None)
    ap.add_argument("--start-epoch", type=int, default=0)
    ap.add_argument("--end-epoch", type=int, default=None,
                    help="Default: full trace (= number of trained epochs).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    exp = Path(args.experiment_dir).resolve()
    if not exp.exists():
        raise FileNotFoundError(f"experiment-dir not found: {exp}")

    targets = args.targets if args.targets else _read_target_list(args.experiment_dir)
    log_dir = exp / "score_lt_iqr_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("LT-IQR", str(log_dir / "score_lt_iqr.log"))
    logger.info(f"experiment: {exp}")
    logger.info(f"targets: {targets}")
    logger.info(f"epoch window: [{args.start_epoch}, {args.end_epoch})")

    with open(exp / "train_config.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    dname = str(train_cfg.get("dataset", {}).get("name", "")).lower()
    arch = train_cfg.get("model", {}).get("architecture", "")
    n_data_points = train_cfg.get("dataset", {}).get("n_data_points", None)

    keep_indices = np.load(exp / "keep_indices.npy")

    runtime_extras = {
        "dataset": dname, "arch": arch,
        "loading": "ram",
        "n_data_points": ("full" if n_data_points is None else int(n_data_points)),
    }

    for tm in targets:
        target_dir = exp / f"model_{int(tm)}"
        loss_p = target_dir / "per_epoch_losses.npy"
        if not loss_p.exists():
            raise FileNotFoundError(
                f"per_epoch_losses.npy missing for target {tm} under {target_dir}. "
                f"Run scripts/train.py --trace-mode loss_only first."
            )
        # On-disk layout is (E, N); transpose to (N, E) for the IQR helper.
        losses_EN = np.load(loss_p).astype(np.float32, copy=False)
        losses_NE = losses_EN.T
        N, E = losses_NE.shape

        keep_eval = keep_indices[int(tm), :N].astype(bool)

        t0 = time.time()
        scores = compute_lt_iqr_scores(
            losses_NE, keep_eval,
            start_epoch=int(args.start_epoch),
            end_epoch=args.end_epoch,
        )
        elapsed = time.time() - t0

        out_p = target_dir / "lt_iqr_scores.npy"
        np.save(out_p, scores)
        logger.debug(
            f"target {tm:>3d}  N={N} E={E} members={int(keep_eval.sum())} "
            f"valid_scores={int(np.isfinite(scores).sum())}  -> {out_p.name} "
            f"({elapsed*1000:.0f} ms)"
        )
        append_runtime(
            str(exp / "runtime.csv"),
            phase="lt_iqr_score", item=f"target_{int(tm)}",
            seconds=elapsed,
            **runtime_extras,
        )


if __name__ == "__main__":
    sys.exit(main())
