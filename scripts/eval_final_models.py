"""Post-hoc evaluation of trained target/shadow models.

For each ``model_<t>/best_model.pth`` (the final-epoch model), loads the
weights and evaluates on the target's own training members (no augmentation)
and on the held-out test set, writing ``model_<t>/metrics.csv`` with:

    Best_epoch, Train_loss, Test_loss, Train_acc (%), Test_acc (%)

Evaluation always uses the dataset-agnostic RAM path (``load_dataset`` +
``create_data_loaders``); the GPU-resident training path is only a speed
optimization over identical data, so the reported losses/accuracies match
the pipeline's in-loop final eval. The per-target member split is read from
``keep_indices.npy`` (and ``subsample_indices.npy`` if the pool was
subsampled, e.g. CINIC-10), mirroring ``scripts/train.py``.

Feeds the target-model quality table (``paper_tabs_figs/tab_model_quality.py``).

Usage:
    python scripts/eval_final_models.py \\
        --config experiments/cifar10/wrn28-2/2025-12-07_0140/train_config.yaml \\
        [--targets 0 25 50 ...]
"""
import _bootstrap  # noqa: F401

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Subset

from rare.utils.common import evaluate, setup_logger
from rare.utils.data import create_data_loaders, load_dataset
from rare.utils.model import get_model

# Paper targets; used when --targets is not given.
DEFAULT_TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]


def _load_pool_datasets(cfg: dict, exp_dir: Path, logger):
    """(train_dataset[aug], test_dataset, per_epoch_eval_dataset[no-aug]) over
    the same (possibly subsampled) train pool that keep_indices indexes."""
    train_dataset, test_dataset, per_epoch_eval_dataset = load_dataset(cfg)
    sub_path = exp_dir / "subsample_indices.npy"
    if sub_path.exists():
        sub = np.load(sub_path)
        if sub.size != len(train_dataset):
            train_dataset = Subset(train_dataset, sub.tolist())
            per_epoch_eval_dataset = Subset(per_epoch_eval_dataset, sub.tolist())
            logger.info(f"[subsample] using {sub.size} samples from {sub_path.name}")
    return train_dataset, test_dataset, per_epoch_eval_dataset


def _evaluate_one(exp_dir: Path, t: int, cfg: dict, keep_row: np.ndarray,
                  train_dataset, test_dataset, per_epoch_eval_dataset,
                  device: torch.device, use_amp: bool, logger) -> Optional[dict]:
    model_dir = exp_dir / f"model_{int(t)}"
    best_p = model_dir / "best_model.pth"
    if not best_p.exists():
        logger.warning(f"[skip] model_{t}: best_model.pth missing")
        return None

    model = get_model(
        int(cfg.get("dataset", {}).get("num_classes", 10)), **cfg.get("model", {})
    ).to(device).eval()
    ckpt = torch.load(best_p, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)

    member_idx = np.where(keep_row)[0]
    shadow_train = Subset(train_dataset, member_idx.tolist())
    shadow_train_eval = Subset(per_epoch_eval_dataset, member_idx.tolist())
    _, test_loader, train_eval_loader, _ = create_data_loaders(
        shadow_train, test_dataset, cfg, shadow_train_eval, per_epoch_eval_dataset,
    )

    criterion = nn.CrossEntropyLoss()
    tr_loss, tr_acc = evaluate(model, train_eval_loader, criterion, device, use_amp=use_amp)
    te_loss, te_acc = evaluate(model, test_loader, criterion, device, use_amp=use_amp)
    best_epoch = ckpt.get("epoch", "N/A") if isinstance(ckpt, dict) else "N/A"

    with (model_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Best_epoch", "Train_loss", "Test_loss", "Train_acc (%)", "Test_acc (%)"])
        w.writerow([best_epoch, f"{tr_loss:.6f}", f"{te_loss:.6f}",
                    f"{tr_acc:.4f}", f"{te_acc:.4f}"])
    logger.info(
        f"model_{t:>3d} | Train {tr_loss:.4f}/{tr_acc:.2f}% "
        f"| Test {te_loss:.4f}/{te_acc:.2f}% -> {model_dir/'metrics.csv'}"
    )
    return {"target": int(t), "train_loss": tr_loss, "train_acc": tr_acc,
            "test_loss": te_loss, "test_acc": te_acc}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=str,
                    help="Path to the experiment's train_config.yaml.")
    ap.add_argument("--targets", type=int, nargs="*", default=None,
                    help=f"Model indices to evaluate (default: {DEFAULT_TARGETS}).")
    args = ap.parse_args()

    with open(Path(args.config).resolve()) as f:
        cfg = yaml.safe_load(f)
    exp_dir = Path(cfg["experiment"]["checkpoint_dir"]).resolve()

    log_dir = exp_dir / "eval_final_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("EvalFinal", str(log_dir / "eval_final.log"))
    logger.info(f"experiment: {exp_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("training", {}).get("use_amp", torch.cuda.is_available()))

    keep = np.load(exp_dir / "keep_indices.npy")
    train_dataset, test_dataset, per_epoch_eval_dataset = _load_pool_datasets(cfg, exp_dir, logger)

    targets: List[int] = list(args.targets) if args.targets is not None else list(DEFAULT_TARGETS)
    written = 0
    for t in targets:
        row = _evaluate_one(
            exp_dir, t, cfg, keep[int(t)].astype(bool),
            train_dataset, test_dataset, per_epoch_eval_dataset,
            device, use_amp, logger,
        )
        written += int(row is not None)
    logger.info(f"Wrote metrics.csv for {written}/{len(targets)} models.")


if __name__ == "__main__":
    main()
