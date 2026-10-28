"""
Train reference reference (reference) models for RARE.

Given an experiment with retrained target models (each having
`init_model.pth` and per-epoch trajectory artefacts) and a saved
`keep_indices.npy`, this script trains M reference shadow models per target and
caches the resulting (loss, margin) trajectories to disk for locked RARE
scoring.

Init policy (when M > 1)
------------------------
- Ref 0 always loads `model_<t>/init_model.pth` — the canonical
  reference: same init as the target, only the data differs (members
  flipped to non-members). Cache tag suffix: `__sharedinit`.
- Refs 1..M-1 use random initialisation seeded by
  `reference_seeds[r]` — independent shadow models trained on the same
  flipped data, providing ensemble diversity. Cache tag suffix:
  `__randinit_seed{S}`. Toggled by `--random-init-after-first` (default
  true).

Per-reference cache filenames live alongside the target trajectory artefacts at
`<exp>/model_<t>/ref_{losses,margin,flip_idx,ref_model}_E{E}_<tag>_ref{r}.{npy,pth}`
so an M=1 run produces the exact same filenames as the historical CIFAR-10
cache (full backward compatibility).

Usage
-----
  python scripts/rare_train_references.py \\
      --experiment-dir experiments/cifar10/wrn28-2/2025-12-07_0140 \\
      --num-references 2 \\
      --reference-seeds 42 43 \\
      --random-init-after-first
"""

import _bootstrap  # noqa: F401

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

from rare.analysis.reference import load_or_train_multiple_references
from rare.utils.common import setup_logger
from rare.utils.data import load_dataset, resolve_subsample
from rare.utils.gpu_dataset import is_gpu_resident_supported, load_image_dataset_to_gpu


def _read_target_list(experiment_dir: str) -> List[int]:
    cfg_path = Path(experiment_dir) / "attack_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"attack_config.yaml not found at {cfg_path}; pass --target-models explicitly"
        )
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "target_models" not in cfg:
        raise KeyError(f"attack_config.yaml has no `target_models`: {cfg_path}")
    return [int(t) for t in cfg["target_models"]]


def _load_raw_dataset(train_cfg: dict, dname: str, data_dir: str):
    """Load the raw (no-transform) training set used by reference training."""
    if dname == "cifar10":
        from torchvision.datasets import CIFAR10
        return CIFAR10(root=data_dir, train=True, download=False, transform=None)
    if dname == "cifar100":
        from torchvision.datasets import CIFAR100
        return CIFAR100(root=data_dir, train=True, download=False, transform=None)
    if dname == "cinic10":
        from torchvision.datasets import ImageFolder
        return ImageFolder(root=os.path.join(data_dir, "cinic10", "train"), transform=None)
    if dname == "gtsrb":
        from torchvision.datasets import GTSRB
        return GTSRB(root=data_dir, split="train", download=False, transform=None)
    # Fallback: use load_dataset's transformed view; only used for unsupported
    # datasets where train-time augs ≈ no-op.
    train_ds, _, _ = load_dataset(train_cfg, mode="train")
    return train_ds


def _load_rare_config(default_path: str = "configs/config_rare.yaml"):
    """Peek at --config, load YAML, return ref_training dict + resolved path."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=default_path)
    pre_args, _ = pre.parse_known_args()
    cfg_path = Path(pre_args.config)
    if not cfg_path.exists():
        return {}, str(cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return (cfg.get("rare", {}) or {}).get("ref_training", {}) or {}, str(cfg_path)


def parse_args() -> argparse.Namespace:
    ref_cfg, cfg_path = _load_rare_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default=cfg_path,
                    help="Path to RARE config YAML. CLI flags override values from this file.")
    ap.add_argument("--experiment-dir", required=True, type=str)
    ap.add_argument("--target-models", type=int, nargs="*", default=None,
                    help="Subset of targets to train references for. Default: every entry in attack_config.yaml.")
    ap.add_argument("--num-references", type=int, default=int(ref_cfg.get("num_references", 1)),
                    help="Number of reference models M per target.")
    ap.add_argument("--reference-seeds", type=int, nargs="*", default=ref_cfg.get("reference_seeds", None),
                    help="Seeds for each ref. Default extends as 42, 43, 44, ... when M>1.")
    ap.add_argument("--cache-tag", type=str, default=str(ref_cfg.get("cache_tag", "nonmem_shadow_sgd")),
                    help="Base cache tag.")
    ap.add_argument("--random-init-after-first", action="store_true", default=bool(ref_cfg.get("random_init_after_first", True)),
                    help="When M>1, refs 1..M-1 use random init seeded by reference_seeds[r].")
    ap.add_argument("--shared-init-all-refs", dest="random_init_after_first", action="store_false",
                    help="Legacy M>1 mode: all refs share target's init (overrides --random-init-after-first).")
    ap.add_argument("--flip-nonmember-frac", type=float, default=float(ref_cfg.get("flip_nonmember_frac", 1.0)))
    ap.add_argument("--flip-nonmember-max", type=int, default=int(ref_cfg.get("flip_nonmember_max", 30000)))
    ap.add_argument("--flip-seed", type=int, default=int(ref_cfg.get("flip_seed", 0)))
    ap.add_argument("--bootstrap-additional-refs", action="store_true",
                    default=bool(ref_cfg.get("bootstrap_additional_refs", False)),
                    help="When set and num_references > 1, train each ref "
                         "ref_idx >= 1 on a bootstrap (sampled with "
                         "replacement) of target's non-members. ref 0 keeps "
                         "the full set. Each bootstrap ref uses its own seed.")
    ap.add_argument("--bootstrap-frac", type=float,
                    default=float(ref_cfg.get("bootstrap_frac", 0.7)),
                    help="Bootstrap sample size = round(frac * |target "
                         "non-members|). Default 0.7 = 70%%.")
    ap.add_argument("--no-load-cache", dest="load_cache", action="store_false",
                    default=bool(ref_cfg.get("load_cache", True)),
                    help="Force re-train references even if cached on disk.")
    ap.add_argument("--no-save-cache", dest="save_cache", action="store_false",
                    default=bool(ref_cfg.get("save_cache", True)))
    ap.add_argument("--no-save-model-checkpoints", dest="save_model_checkpoints",
                    action="store_false", default=bool(ref_cfg.get("save_model_checkpoints", True)))
    ap.add_argument("--strict-require-all-refs", action="store_true",
                    default=bool(ref_cfg.get("strict_require_all_refs", False)))
    ap.add_argument("--auto-repair-missing-ckpt", action="store_true",
                    default=bool(ref_cfg.get("auto_repair_missing_ckpt", True)))
    ap.add_argument("--shadow-batch-size", type=int, default=int(ref_cfg.get("shadow_batch_size", 256)))
    ap.add_argument("--shadow-num-workers", type=int, default=int(ref_cfg.get("shadow_num_workers", 4)))
    ap.add_argument("--eval-batch-size", type=int, default=int(ref_cfg.get("eval_batch_size", 4096)))
    ap.add_argument("--device", type=str, default=str(ref_cfg.get("device", "cuda")))
    ap.add_argument("--show-progress", action="store_true", default=bool(ref_cfg.get("show_progress", True)))
    ap.add_argument("--save-reference-logits", action="store_true", default=True,
                    help="After training, cache each reference's logits "
                         "(ref_logits_*.npy) for the logit-replacement defense.")
    ap.add_argument("--no-save-reference-logits", dest="save_reference_logits",
                    action="store_false")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    exp = Path(args.experiment_dir).resolve()
    if not exp.exists():
        raise FileNotFoundError(f"experiment-dir not found: {exp}")

    targets = args.target_models if args.target_models else _read_target_list(args.experiment_dir)

    log_dir = exp / "rare_train_ref_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("RARE-Ref", str(log_dir / "rare_train_references.log"))
    logger.info(f"experiment: {exp}")
    logger.info(f"targets: {targets}")
    logger.info(f"num_references (M): {args.num_references}")

    M = int(args.num_references)
    if args.reference_seeds is None:
        ref_seeds = [42 + r for r in range(M)]
    else:
        ref_seeds = list(args.reference_seeds)[:M]
        while len(ref_seeds) < M:
            ref_seeds.append(42 + len(ref_seeds))
    logger.info(f"reference_seeds: {ref_seeds}")
    logger.info(f"random_init_after_first: {args.random_init_after_first}")

    # Load configs
    with open(exp / "train_config.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    ds_cfg = train_cfg.get("dataset", {})
    num_classes = int(ds_cfg.get("num_classes", 10))
    dname = str(ds_cfg["name"]).lower()
    data_dir = ds_cfg.get("data_dir", "data")
    n_data_points = ds_cfg.get("n_data_points", None)
    subsample_seed = int(ds_cfg.get("subsample_seed", train_cfg.get("seed", 42)))
    loading_cfg = str(ds_cfg.get("loading", "gpu")).lower()
    n_epochs = int(train_cfg.get("training", {}).get("epochs", 100))

    # Raw dataset
    train_dataset_raw = _load_raw_dataset(train_cfg, dname, data_dir)
    n_full_pool = len(train_dataset_raw)
    logger.info(f"dataset={dname}, raw_pool={n_full_pool}, n_data_points={n_data_points}")

    # Apply experiment's deterministic subsample
    sub_idx = resolve_subsample(
        n_full_pool, n_data_points, subsample_seed,
        save_path=str(exp / "subsample_indices.npy"),
    )
    if sub_idx.size != n_full_pool:
        from torch.utils.data import Subset as _Subset
        train_dataset_raw = _Subset(train_dataset_raw, sub_idx.tolist())
        logger.info(f"applied subsample: {sub_idx.size}/{n_full_pool} (seed={subsample_seed})")

    # GPU-resident data (for fast reference training)
    use_gpu = (
        loading_cfg == "gpu"
        and torch.cuda.is_available()
        and str(args.device) == "cuda"
        and is_gpu_resident_supported(dname)
    )
    gpu_resident_data = None
    if use_gpu:
        logger.info(f"[gpu-resident] pre-loading {dname} to GPU...")
        gimg, glab, gmean, gstd = load_image_dataset_to_gpu(
            dname, data_dir, train=True, device=torch.device("cuda"),
        )
        if sub_idx.size != int(gimg.shape[0]):
            sub_idx_t = torch.as_tensor(sub_idx, device=gimg.device, dtype=torch.long)
            gimg = gimg[sub_idx_t].contiguous()
            glab = glab[sub_idx_t].contiguous()
        gpu_resident_data = {"images_uint8": gimg, "labels": glab, "mean": gmean, "std": gstd}
        logger.info(
            f"[gpu-resident] images={tuple(gimg.shape)} "
            f"({gimg.numel() / 1024 / 1024:.0f} MB), dtype={gimg.dtype}"
        )

    # Load keep_indices
    keep_path = exp / "keep_indices.npy"
    if not keep_path.exists():
        raise FileNotFoundError(f"keep_indices.npy missing: {keep_path}")
    keep_indices = np.load(keep_path)
    logger.info(f"keep_indices: {keep_indices.shape}")

    # Train references per target.
    arch = train_cfg.get("model", {}).get("architecture", "")
    runtime_extras = {
        "dataset": dname,
        "arch": arch,
        "loading": ("gpu" if use_gpu else "ram"),
        "n_data_points": ("full" if n_data_points is None else int(n_data_points)),
    }

    overall_t0 = time.time()
    for tm in targets:
        target_init_path = exp / f"model_{int(tm)}" / "init_model.pth"
        if not target_init_path.exists():
            logger.warning(
                f"model_{int(tm)}/init_model.pth missing — reference 0 will use random init "
                f"(loses shared-init reference semantics). Run scripts/train.py first "
                f"to save init."
            )

        keep_eval = keep_indices[int(tm)].astype(bool)
        logger.info(
            f"\n========== target {tm} (members={keep_eval.sum()}, "
            f"non-members={(~keep_eval).sum()}) =========="
        )

        ref_losses, flip_idx, full_tags, ref_seeds_used = (
            load_or_train_multiple_references(
                experiment_dir=str(exp),
                target_model=int(tm),
                config=train_cfg,
                train_dataset_raw=train_dataset_raw,
                membership_mask=keep_eval,
                num_classes=num_classes,
                device=args.device,
                n_epochs=n_epochs,
                cache_tag=str(args.cache_tag),
                num_references=M,
                reference_seeds=ref_seeds,
                load_if_exists=bool(args.load_cache),
                save_cache=bool(args.save_cache),
                show_progress=bool(args.show_progress),
                flip_nonmember_frac=float(args.flip_nonmember_frac),
                flip_nonmember_max=int(args.flip_nonmember_max),
                flip_seed=int(args.flip_seed),
                eval_batch_size=int(args.eval_batch_size),
                shadow_batch_size=int(args.shadow_batch_size),
                shadow_num_workers=int(args.shadow_num_workers),
                save_model_checkpoints=bool(args.save_model_checkpoints),
                auto_repair_missing_ckpt=bool(args.auto_repair_missing_ckpt),
                strict_require_all_refs=bool(args.strict_require_all_refs),
                rare_models_dir=str(exp),
                gpu_resident_data=gpu_resident_data,
                amp_dtype="bf16",
                init_state_path=str(target_init_path) if target_init_path.exists() else None,
                runtime_csv=str(exp / "runtime.csv"),
                runtime_extras={**runtime_extras, "target_model": int(tm)},
                random_init_after_first=bool(args.random_init_after_first),
                bootstrap_additional_refs=bool(args.bootstrap_additional_refs),
                bootstrap_frac=float(args.bootstrap_frac),
            )
        )

        # Per-target tag bookkeeping. For M=1 we record the single tag; for
        # M>1 each ref has its own. The downstream scorer reads the tag list
        # by re-deriving it (same formula), so we just log here.
        logger.debug(f"target {tm} reference tags: {full_tags}")

    elapsed = time.time() - overall_t0
    logger.info(f"\nAll references trained in {elapsed / 60:.1f} min")

    # Cache reference logits for the logit-replacement defense, on the same
    # basis as the attacks' logits.npy (A=2, experiment inference_mode). Reads
    # the experiment's logit_config.yaml; skipped if absent or disabled.
    if bool(args.save_reference_logits):
        logit_cfg_path = exp / "logit_config.yaml"
        if not logit_cfg_path.exists():
            logger.warning(
                f"save_reference_logits: {logit_cfg_path} not found; skipping "
                "(run generate_logits.py first, or pass --no-save-reference-logits)."
            )
        else:
            from rare.analysis.reference import generate_reference_logits
            logger.info("\nCaching reference logits for the defense...")
            generate_reference_logits(
                experiment_dir=str(exp),
                targets=[int(t) for t in targets],
                logit_config=yaml.safe_load(open(logit_cfg_path)),
                num_refs=int(M),
                logger=logger,
                overwrite=False,
            )


if __name__ == "__main__":
    sys.exit(main())
