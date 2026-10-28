"""Train shadow / target models for membership-inference experiments.

Unified entrypoint for both:
  (a) Shadow training over `[start_shadow_model_idx, end_shadow_model_idx]`
      (default; no per-epoch tracing). Runtime phase = `train`.
  (b) Target retraining over `target_models` (top-level config field, or
      `--targets <i> ...` CLI override) with one of three per-epoch trace
      modes selected by `--trace-mode`:
          none      : no traces. Runtime phase = `target_train_baseline`.
                      Used to measure the baseline target-retrain wall time
                      reported in the runtime table.
          loss_only : per_epoch_losses.npy. Used by LT-IQR / Liu-MLP / LOSS.
                      Runtime phase = `target_train_loss_trace`.
          full      : per_epoch_losses.npy + per_epoch_margin.npy. Used by
                      RARE. Runtime phase = `target_train_full_trace`.

  When `--trace-mode != none`, `training.{eval_per_epoch, save_models,
  early_stopping}` are forced off so the run is purely a per-epoch trace
  dump on a fixed final-epoch model.

Two execution paths chosen by the config field `dataset.loading`:

  - 'gpu': the entire train/test pool lives in GPU memory; augmentation runs
    via vectorised CUDA ops (see `rare/utils/gpu_dataset.py`). Recommended for
    CIFAR-shaped datasets that fit in <1 GB. Removes the CPU DataLoader
    bottleneck and gives a 3-7x epoch-time speedup on Blackwell-class GPUs.

  - 'ram': classic torchvision/LMDB CPU DataLoader path. Use for CPU-only
    debug runs or for datasets without a GPU-resident loader yet.

The `dataset.n_data_points` field (default null = full pool) lets you train
on a deterministic random subsample of the train pool — used to match the
LT-IQR / RMIA CINIC-10 50k setup. Subsample indices are persisted to
`subsample_indices.npy` in the experiment dir; subsequent runs reuse them.

Per-model wall time is appended to `<exp>/runtime.csv` under the phase name
listed above.
"""

import _bootstrap  # noqa: F401  -- adds repo root to sys.path

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from rare.utils.common import (
    PhaseTimer,
    aggregate_shadow_metrics,
    cleanup_gpu_memory,
    parse_overrides,
    recursive_update,
    save_config,
    set_seed,
    setup_logger,
    validate_config,
)
from rare.utils.data import (
    create_data_loaders,
    load_dataset,
    resolve_subsample,
)
from rare.utils.gpu_dataset import (
    create_gpu_resident_loaders,
    is_gpu_resident_supported,
    load_image_dataset_to_gpu,
)
from rare.utils.model import get_model
from rare.utils.train import train_model


def setup_experiment(config):
    """Create exp dir, set up logging/seeding, return (dir, logger, device)."""
    dataset = config.get('dataset', {}).get('name', 'cifar10')
    arch = config.get('model', {}).get('architecture', 'resnet18')
    if config.get('experiment', {}).get('checkpoint_dir', 'none') == 'none':
        experiment_dir = os.path.join(
            'experiments', dataset, arch,
            datetime.now().strftime('%Y-%m-%d_%H%M'),
        )
        os.makedirs(experiment_dir, exist_ok=True)
        config.setdefault('experiment', {})
        config['experiment']['checkpoint_dir'] = experiment_dir
    else:
        experiment_dir = config['experiment']['checkpoint_dir']
        os.makedirs(experiment_dir, exist_ok=True)

    log_level_str = config.get('experiment', {}).get('log_level', 'info').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger(
        'Train',
        os.path.join(experiment_dir, 'train_log.log'),
        level=log_level,
    )

    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    use_cuda = config.get('use_cuda', True)
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    save_config(config, os.path.join(experiment_dir, 'train_config.yaml'))
    logger.info(f"Saved configuration to {experiment_dir}/train_config.yaml")
    return experiment_dir, logger, device


def _resolve_loading_mode(config, device, dataset_name, logger):
    """Pick 'gpu' or 'ram' from config + capability checks. Falls back to 'ram'
    (with a warning) if 'gpu' is requested but unsupported for this run.
    """
    loading = str(config.get('dataset', {}).get('loading', 'gpu')).lower()
    if loading not in ('gpu', 'ram'):
        raise ValueError(
            f"dataset.loading must be 'gpu' or 'ram'; got {loading!r}"
        )
    if loading == 'gpu':
        if device.type != 'cuda':
            logger.warning("dataset.loading='gpu' requested but no CUDA device — falling back to 'ram'.")
            return 'ram'
        if not is_gpu_resident_supported(dataset_name):
            logger.warning(
                f"dataset.loading='gpu' requested but {dataset_name!r} has no GPU-resident "
                "loader — falling back to 'ram'."
            )
            return 'ram'
    return loading


def _save_or_load_init_state(model, shadow_dir, device, logger):
    """Persist (or restore) the freshly-constructed init weights for this shadow.

    On the first run for a given index, the seeded random init is saved.
    On any re-run — whether mid-training-crash recovery, a `--trace-mode`
    target retrain, or `--overwrite` — the saved init is loaded back so any
    retrain starts from byte-identical weights to the original shadow.

    Combined with the matching per-target seed convention
    (`seed = config.seed + i + 1`, used uniformly across this script), this
    guarantees that retraining target index `t` reproduces the same training
    trajectory as the original shadow, so `best_model.pth` stays consistent
    with the LiRA / RMIA / Attack-R score arrays that were computed against
    it.
    """
    path = os.path.join(shadow_dir, 'init_model.pth')
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        logger.info(f"  loaded init weights <- {path}")
    else:
        torch.save(model.state_dict(), path)
        logger.info(f"  saved init weights -> {path}")


def _get_or_build_keep_indices(experiment_dir, n_shadows, n_train, pkeep, seed, logger):
    """Reuse keep_indices.npy if present; otherwise generate + persist."""
    path = os.path.join(experiment_dir, 'keep_indices.npy')
    if os.path.exists(path):
        keep = np.load(path)
        logger.info(f"Loaded keep_indices from {path} shape={keep.shape}")
        if keep.shape[1] != n_train:
            raise ValueError(
                f"keep_indices.shape[1]={keep.shape[1]} disagrees with train pool size "
                f"{n_train}. Delete {path} or rebuild for the new pool size."
            )
        return keep
    rng = np.random.default_rng(int(seed))
    keep = rng.uniform(size=(int(n_shadows), int(n_train))) < float(pkeep)
    np.save(path, keep)
    logger.info(
        f"Wrote fresh keep_indices to {path} shape={keep.shape} pkeep={pkeep}"
    )
    return keep


# =============================================================================
# RAM (CPU DataLoader) path
# =============================================================================
def _train_one_shadow_ram(
    config, train_dataset, test_dataset, per_epoch_eval_dataset,
    keep_row, track_losses, track_margin, device, shadow_dir, logger,
    overwrite_checkpoint=True,
):
    train_idxs = np.arange(len(train_dataset))
    member_idx = train_idxs[keep_row]
    shadow_train = Subset(train_dataset, member_idx)
    shadow_train_eval = Subset(per_epoch_eval_dataset, member_idx)
    logger.info(f"Shadow train dataset size: {len(shadow_train)}")

    train_loader, test_loader, train_eval_loader, per_epoch_eval_loader = create_data_loaders(
        shadow_train, test_dataset, config, shadow_train_eval, per_epoch_eval_dataset,
    )
    num_classes = config.get('dataset', {}).get('num_classes', 10)
    model = get_model(num_classes, **config.get('model', {})).to(device)
    _save_or_load_init_state(model, shadow_dir, device, logger)

    model = train_model(
        model, train_loader, test_loader, train_eval_loader, per_epoch_eval_loader,
        config, device, save_dir=shadow_dir, logger=logger,
        track_losses=track_losses, track_margin=track_margin,
        overwrite_checkpoint=overwrite_checkpoint,
    )
    del model, train_loader, test_loader, train_eval_loader, per_epoch_eval_loader
    cleanup_gpu_memory()


# =============================================================================
# GPU-resident path
# =============================================================================
def _train_one_shadow_gpu(
    config, train_images_gpu, train_labels_gpu, test_images_gpu, test_labels_gpu,
    mean_gpu, std_gpu, keep_row, track_losses, track_margin, device, shadow_dir, logger,
    pad_mode, amp_dtype, overwrite_checkpoint=True,
):
    member_idx = np.where(keep_row)[0]
    logger.info(
        f"Shadow members={member_idx.size}/{int(train_images_gpu.shape[0])} "
        f"({100.0 * member_idx.size / int(train_images_gpu.shape[0]):.1f}%)"
    )

    train_loader, test_loader, train_eval_loader, per_epoch_eval_loader = (
        create_gpu_resident_loaders(
            train_images_uint8=train_images_gpu,
            train_labels=train_labels_gpu,
            test_images_uint8=test_images_gpu,
            test_labels=test_labels_gpu,
            mean=mean_gpu,
            std=std_gpu,
            member_indices=member_idx,
            train_cfg=config,
            device=device,
            pad_mode=pad_mode,
        )
    )

    num_classes = int(config.get('dataset', {}).get('num_classes', 10))
    model = get_model(num_classes, **config.get('model', {})).to(device)
    _save_or_load_init_state(model, shadow_dir, device, logger)

    train_model(
        model, train_loader, test_loader, train_eval_loader, per_epoch_eval_loader,
        config, device, save_dir=shadow_dir, logger=logger,
        track_losses=track_losses, track_margin=track_margin,
        amp_dtype=amp_dtype,
        overwrite_checkpoint=overwrite_checkpoint,
    )

    # Alias best_model.pth -> checkpoint.pth so the rest of the pipeline finds it.
    best_path = os.path.join(shadow_dir, "best_model.pth")
    ckpt_path = os.path.join(shadow_dir, "checkpoint.pth")
    if os.path.exists(best_path) and not os.path.exists(ckpt_path):
        try:
            os.symlink(os.path.basename(best_path), ckpt_path)
        except OSError:
            import shutil
            shutil.copyfile(best_path, ckpt_path)

    del model, train_loader, test_loader, train_eval_loader, per_epoch_eval_loader
    cleanup_gpu_memory()


def main():
    parser = argparse.ArgumentParser(description='Train shadow / target models for MIA experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument(
        '--override', nargs='*', default=[],
        help='Override config values. Format: key1.subkey=value key2=value',
    )
    parser.add_argument(
        '--pad-mode', type=str, default='reflect', choices=['reflect', 'constant'],
        help="GPU-resident RandomCrop padding mode. Ignored on the RAM path.",
    )
    parser.add_argument(
        '--amp-dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
        help="GPU-resident autocast dtype. Ignored on the RAM path.",
    )
    parser.add_argument(
        '--trace-mode', type=str, default='none',
        choices=['none', 'loss_only', 'full'],
        help=(
            "Per-epoch trace mode (default: none). "
            "loss_only writes per_epoch_losses.npy (LT-IQR/Liu-MLP/LOSS). "
            "full writes per_epoch_{losses,margin}.npy (RARE). "
            "When != 'none', training.{eval_per_epoch, save_models, early_stopping} "
            "are forced off."
        ),
    )
    parser.add_argument(
        '--targets', type=int, nargs='*', default=None,
        help=(
            "Optional list of model indices to (re)train. Overrides the "
            "top-level 'target_models' config field. When --trace-mode != none, "
            "must resolve to a non-empty list."
        ),
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Bypass the skip-when-best_model.pth-exists check.',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    overrides = parse_overrides(args.override)
    recursive_update(config, overrides)

    required_keys = [
        'dataset.name', 'training.num_shadow_models', 'training.epochs',
        'training.batch_size', 'model.architecture',
    ]
    try:
        validate_config(config, required_keys)
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)

    experiment_dir, logger, device = setup_experiment(config)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass

    ds_cfg = config.get('dataset', {})
    dname = str(ds_cfg.get('name', '')).lower()
    n_data_points = ds_cfg.get('n_data_points', None)
    subsample_seed = int(ds_cfg.get('subsample_seed', config.get('seed', 42)))

    loading_mode = _resolve_loading_mode(config, device, dname, logger)
    logger.info(f"Loading mode: {loading_mode}  (config dataset.loading={ds_cfg.get('loading', 'gpu')!r})")

    t_cfg = config.get('training', {})
    n_shadows_full = int(t_cfg.get('num_shadow_models', 1))
    pkeep = float(ds_cfg.get('pkeep', 0.5))
    start_idx = int(t_cfg.get('start_shadow_model_idx', 0))
    end_idx = int(t_cfg.get('end_shadow_model_idx', n_shadows_full - 1))

    trace_mode = str(args.trace_mode).lower()
    track_losses_flag = trace_mode in ('loss_only', 'full')
    track_margin_flag = trace_mode == 'full'

    # When tracing, the run is purely a per-epoch trace dump on the
    # final-epoch model — no per-epoch test eval, no extra checkpoints,
    # no best-model selection.
    if trace_mode != 'none':
        config.setdefault('training', {})
        config['training']['eval_per_epoch'] = False
        config['training']['save_models'] = False
        config['training']['early_stopping'] = False

    # Resolve the index list. config.target_models is only consulted when we
    # are explicitly in a target-retrain context (--trace-mode != none); the
    # default shadow run trains the full shadow range regardless of whether
    # target_models is set, so a single shared config can drive both.
    target_models_cfg = config.get('target_models', None) or []
    if isinstance(target_models_cfg, str) and target_models_cfg.lower() == 'none':
        target_models_cfg = []
    if args.targets is not None:
        indices = [int(t) for t in args.targets]
        index_source = '--targets'
    elif trace_mode != 'none':
        if not target_models_cfg:
            raise ValueError(
                "--trace-mode != 'none' requires either --targets <i> ... or "
                "a top-level 'target_models' list in the config."
            )
        indices = [int(t) for t in target_models_cfg]
        index_source = "config 'target_models'"
    else:
        indices = list(range(start_idx, end_idx + 1))
        index_source = f'shadow range [{start_idx}, {end_idx}]'
    is_target_run = (trace_mode != 'none') or (args.targets is not None)

    phase_name = (
        'train' if not is_target_run else {
            'none':      'target_train_baseline',
            'loss_only': 'target_train_loss_trace',
            'full':      'target_train_full_trace',
        }[trace_mode]
    )

    runtime_csv = os.path.join(experiment_dir, 'runtime.csv')
    runtime_extras = {
        'dataset': dname, 'arch': config.get('model', {}).get('architecture', ''),
        'loading': loading_mode,
        'n_data_points': 'full' if n_data_points is None else int(n_data_points),
    }

    # -------- per-loading-mode prep -----------------------------------------
    if loading_mode == 'gpu':
        data_dir = ds_cfg.get('data_dir', 'data')
        logger.info(f"[gpu] pre-loading {dname} train+test...")
        train_images_gpu, train_labels_gpu, mean_gpu, std_gpu = load_image_dataset_to_gpu(
            dname, data_dir, train=True, device=device,
        )
        test_images_gpu, test_labels_gpu, _, _ = load_image_dataset_to_gpu(
            dname, data_dir, train=False, device=device,
        )

        n_full = int(train_images_gpu.shape[0])
        sub_idx = resolve_subsample(
            n_full, n_data_points, subsample_seed,
            save_path=os.path.join(experiment_dir, 'subsample_indices.npy'),
        )
        if sub_idx.size != n_full:
            sub_idx_gpu = torch.as_tensor(sub_idx, device=device, dtype=torch.long)
            train_images_gpu = train_images_gpu[sub_idx_gpu].contiguous()
            train_labels_gpu = train_labels_gpu[sub_idx_gpu].contiguous()
            logger.info(
                f"[subsample] {sub_idx.size}/{n_full} samples (seed={subsample_seed}) "
                f"persisted to {os.path.join(experiment_dir, 'subsample_indices.npy')}"
            )
        n_train = int(train_images_gpu.shape[0])
        logger.info(
            f"[gpu] train={tuple(train_images_gpu.shape)}, "
            f"test={tuple(test_images_gpu.shape)}"
        )
    else:
        logger.info("Loading dataset...")
        train_dataset, test_dataset, per_epoch_eval_dataset = load_dataset(config)
        n_full = len(train_dataset)
        sub_idx = resolve_subsample(
            n_full, n_data_points, subsample_seed,
            save_path=os.path.join(experiment_dir, 'subsample_indices.npy'),
        )
        if sub_idx.size != n_full:
            train_dataset = Subset(train_dataset, sub_idx.tolist())
            per_epoch_eval_dataset = Subset(per_epoch_eval_dataset, sub_idx.tolist())
            logger.info(
                f"[subsample] {sub_idx.size}/{n_full} samples (seed={subsample_seed}) "
                f"persisted to {os.path.join(experiment_dir, 'subsample_indices.npy')}"
            )
        n_train = len(train_dataset)
        logger.info(f"train_dataset size: {n_train}, test_dataset size: {len(test_dataset)}")

    # -------- keep_indices over the (possibly subsampled) train pool --------
    keep_indices = _get_or_build_keep_indices(
        experiment_dir, n_shadows_full, n_train, pkeep, config.get('seed', 42), logger,
    )

    role = 'target' if is_target_run else 'shadow'
    logger.info(
        f"Training {len(indices)} {role} model(s) from {index_source} "
        f"(trace_mode={trace_mode}, phase={phase_name}, overwrite={args.overwrite})"
    )

    # -------- main training loop --------------------------------------------
    for i in indices:
        shadow_dir = os.path.join(experiment_dir, f'model_{i}')
        os.makedirs(shadow_dir, exist_ok=True)
        ckpt_path = os.path.join(shadow_dir, 'checkpoint.pth')
        best_path = os.path.join(shadow_dir, 'best_model.pth')
        loss_trace_path = os.path.join(shadow_dir, 'per_epoch_losses.npy')
        margin_trace_path = os.path.join(shadow_dir, 'per_epoch_margin.npy')

        # Skip-if-complete. For trace runs we also require the per-epoch
        # artefacts so re-invoking with --trace-mode full upgrades a prior
        # loss_only run instead of being a no-op.
        has_ckpt = os.path.exists(ckpt_path) or os.path.exists(best_path)
        required_traces = []
        if track_losses_flag:
            required_traces.append(loss_trace_path)
        if track_margin_flag:
            required_traces.append(margin_trace_path)
        has_traces = all(os.path.exists(p) for p in required_traces)
        if has_ckpt and has_traces and not args.overwrite:
            marker = 'checkpoint.pth' if os.path.exists(ckpt_path) else 'best_model.pth'
            extra = ' + per_epoch traces' if required_traces else ''
            logger.info(f"[skip] {role} {i}: {marker}{extra} present (--overwrite to redo)")
            continue

        # Match per-shadow seed exactly so target retrains reproduce the
        # original shadow's trajectory.
        seed_i = int(config.get('seed', 42)) + int(i) + 1
        set_seed(seed_i)

        # Trace retrains preserve any pre-existing best_model.pth so the logits
        # generated against the original target weights remain valid. Only the
        # per_epoch trace .npy files are (re)written. Plain training (shadow
        # run or target baseline retrain) keeps the normal save behaviour.
        overwrite_ckpt = trace_mode == 'none'

        with PhaseTimer(runtime_csv, phase=phase_name, item=f'{role}_{i}', **runtime_extras):
            if loading_mode == 'gpu':
                _train_one_shadow_gpu(
                    config, train_images_gpu, train_labels_gpu,
                    test_images_gpu, test_labels_gpu, mean_gpu, std_gpu,
                    keep_indices[i], track_losses_flag, track_margin_flag,
                    device, shadow_dir, logger,
                    pad_mode=args.pad_mode, amp_dtype=args.amp_dtype,
                    overwrite_checkpoint=overwrite_ckpt,
                )
            else:
                _train_one_shadow_ram(
                    config, train_dataset, test_dataset, per_epoch_eval_dataset,
                    keep_indices[i], track_losses_flag, track_margin_flag,
                    device, shadow_dir, logger,
                    overwrite_checkpoint=overwrite_ckpt,
                )
        logger.info(f"{role.capitalize()} model {i} trained -> {shadow_dir}")

    if is_target_run:
        logger.info(f"All {role} retrains ({trace_mode}) complete.")
    else:
        logger.info("All shadow models training completed successfully")
        aggregate_shadow_metrics(experiment_dir, n_shadows_full, logger=logger)


if __name__ == '__main__':
    main()
