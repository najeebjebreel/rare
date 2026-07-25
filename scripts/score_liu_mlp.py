"""
Compute Liu-MLP vulnerability scores for each target model in a benchmark
using cross-target leave-one-out training over the per-epoch losses saved
by the RARE retrain (model_<t>/per_epoch_losses.npy).

Reads and writes under model_<t>/ so this baseline scores from the same
retrained target trajectory RARE consumes — apples-to-apples comparison
per EXPERIMENTS.md §8.

Outputs (per target):
  experiments/<bench>/<run>/model_<t>/liu_mlp_scores_R{N}.npy
  experiments/<bench>/<run>/model_<t>/liu_mlp_labels_R{N}.npy

where N = (pool size) - 1 is the number of labelled shadow trajectories used
for LOO training. The R suffix is derived automatically from the size of the
`--target-models` pool passed in.

Prerequisite:
  Run scripts/train.py --trace-mode loss_only (or full) first so
  per_epoch_losses.npy is populated under each model_<t>/.

Pool coverage strategy
----------------------
Each invocation runs cross-target LOO within ONE pool: each pool member is
scored by training the MLP on the other (pool_size - 1) members. To report
Liu-MLP at multiple R budgets across all 10 targets, you must invoke this
script multiple times so that every target appears in at least one pool of
size R+1 for each desired R.

For the standard 10-target setup [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]:

  # R=1 (pool size 2) — 5 disjoint pairs perfectly cover all 10 targets
  python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 0 25
  python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 50 75
  python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 100 125
  python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 150 175
  python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 200 250

  # R=2 (pool size 3) — 10 is not divisible by 3; use 3 disjoint triples plus
  # one filler pool with --score-only so the filler doesn't overwrite the
  # already-scored 0, 25 R=2 scores.
  python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 0 25 50
  python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 75 100 125
  python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 150 175 200
  python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 0 25 250 --score-only 250

  # R=4 (pool size 5) — 2 disjoint pools of 5 perfectly cover all 10 targets
  python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 0 25 50 75 100
  python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 125 150 175 200 250

Total: 11 invocations × ~30-60 s each = ~10 min for full coverage at R∈{1,2,4}.

Usage:
  python scripts/score_liu_mlp.py \\
      --experiment-dir experiments/cifar10/wrn28-2/2025-12-07_0140 \\
      --target-models 0 25 \\
      --device cuda
"""

import _bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import yaml

from rare.analysis.liu_mlp import load_membership, score_liu_mlp
from rare.utils.common import append_runtime


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", required=True, type=str)
    ap.add_argument(
        "--target-models",
        type=int,
        nargs="*",
        default=None,
        help="LOO pool of target indices used for both training and evaluation; "
             "default reads attack_config.yaml `target_models`.",
    )
    ap.add_argument(
        "--score-only",
        type=int,
        nargs="*",
        default=None,
        help="Subset of `--target-models` to actually score this run (e.g. for smoke tests). "
             "Training still uses the LOO pool minus the held-out test target.",
    )
    ap.add_argument("--n-data-points", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    # Defaults are the reported LT-MLP config (AdamW+cosine, 20 ep, 256-128-64,
    # dropout 0.2, standardized). Override to reproduce the release recipe:
    #   --optimizer sgd --hidden-dims 512 128 32 --dropout 0.0 --epochs 100 \
    #   --batch-size 128 --lr 1e-2 --weight-decay 1e-4 --no-standardize
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--no-standardize", dest="standardize", action="store_false",
                    help="Feed raw trajectories (the release recipe) instead of "
                         "the default train-only robust (median/IQR) scaler.")
    ap.set_defaults(standardize=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def _read_target_list(experiment_dir: str) -> List[int]:
    cfg_path = Path(experiment_dir) / "attack_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"attack_config.yaml not found at {cfg_path}; pass --target-models explicitly")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "target_models" not in cfg:
        raise KeyError(f"attack_config.yaml has no `target_models` key: {cfg_path}")
    return [int(t) for t in cfg["target_models"]]


def main() -> None:
    args = parse_args()
    targets = args.target_models if args.target_models else _read_target_list(args.experiment_dir)
    if args.score_only:
        unknown = [t for t in args.score_only if t not in targets]
        if unknown:
            raise ValueError(f"--score-only contains targets outside the LOO pool: {unknown}")
        score_targets = list(args.score_only)
    else:
        score_targets = list(targets)

    print(f"[liu_mlp] benchmark={args.experiment_dir}")
    print(f"[liu_mlp] LOO pool targets={targets}")
    print(f"[liu_mlp] scoring targets={score_targets}")
    print(f"[liu_mlp] device={args.device}, epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}")

    # The labelled-shadow budget for each scored target = pool size - 1
    # (the LOO training set sees every target other than the one being scored).
    # We persist with explicit `_R{N}` filename suffix so multiple budgets
    # can coexist on disk and be selected at analysis time.
    R_loo = max(1, len(targets) - 1)
    suffix = f"_R{R_loo}"

    runtime_csv = str(Path(args.experiment_dir) / "runtime.csv")

    # Read dataset/arch metadata from train_config so the runtime row is self-describing.
    train_cfg_path = Path(args.experiment_dir) / "train_config.yaml"
    runtime_extras: dict = {}
    if train_cfg_path.exists():
        with open(train_cfg_path) as f:
            tcfg = yaml.safe_load(f) or {}
        ds = tcfg.get("dataset", {})
        runtime_extras = {
            "dataset": str(ds.get("name", "")).lower(),
            "arch": str(tcfg.get("model", {}).get("architecture", "")),
            "loading": str(ds.get("loading", "gpu")).lower(),
            "n_data_points": ("full" if ds.get("n_data_points") is None else int(ds.get("n_data_points"))),
        }

    for t in score_targets:
        out_dir = Path(args.experiment_dir) / f"model_{t}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sp = out_dir / f"liu_mlp_scores{suffix}.npy"
        lp = out_dir / f"liu_mlp_labels{suffix}.npy"

        if sp.exists() and lp.exists() and not args.overwrite:
            print(f"  target {t}: scores already exist; skip ({sp.name}). Use --overwrite to recompute.")
            continue

        print(f"\n[liu_mlp] === target {t}  (LOO over {len(targets)} targets, R={R_loo} labelled shadows) ===")
        t0 = time.perf_counter()
        scores = score_liu_mlp(
            experiment_dir=args.experiment_dir,
            test_target=int(t),
            target_models=targets,
            device=args.device,
            n_data_points=args.n_data_points,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            optimizer=args.optimizer,
            standardize=args.standardize,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - t0
        labels = load_membership(args.experiment_dir, int(t), n_data_points=args.n_data_points)

        np.save(sp, scores.astype(np.float32, copy=False))
        np.save(lp, labels.astype(np.int8, copy=False))
        print(
            f"  saved {sp.name}, {lp.name}  "
            f"| scores shape={scores.shape}  member-mean={scores[labels.astype(bool)].mean():.4f}  "
            f"non-member-mean={scores[~labels.astype(bool)].mean():.4f}"
        )
        append_runtime(
            runtime_csv, phase="liu_mlp_score", item=f"target_{int(t)}",
            seconds=elapsed, R=int(R_loo), **runtime_extras,
        )


if __name__ == "__main__":
    main()
