"""
Reference-model training for vulnerability analysis.

Trains shadow-style models on the non-member subset of a target model's
training data, then evaluates per-epoch (loss, margin) trajectories on the
held-in members and an optional "flip" subset of non-members. RARE reference
artifacts are cached under `<experiment_dir>/model_<target>/` next to the
target traces.

Margin per (sample, epoch) = z[true_class] - max_{c != true_class} z[c].
Current locked RARE uses the reference loss trajectories for LT-BTM-R.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from rare.analysis.utils import IndexedDataset, ProgressBar
from rare.utils.data import TransformSubset, build_transforms
from rare.utils.model import get_model
from rare.utils.train import get_optimizer, get_scheduler


# =============================================================================
# Cache paths (under rare_models_dir/model_<tm>/; callers pass experiment_dir
# as rare_models_dir in the locked pipeline).
# =============================================================================
def _rare_target_dir(rare_models_dir: Path, tm: int) -> Path:
    return Path(rare_models_dir) / f"model_{int(tm)}"


def _ref_losses_path(rare_dir: Path, tm: int, E: int, tag: str, ref_idx: int) -> Path:
    return _rare_target_dir(rare_dir, tm) / f"ref_losses_E{E}_{tag}_ref{ref_idx}.npy"


def _ref_margin_path(rare_dir: Path, tm: int, E: int, tag: str, ref_idx: int) -> Path:
    return _rare_target_dir(rare_dir, tm) / f"ref_margin_E{E}_{tag}_ref{ref_idx}.npy"


def _ref_flip_path(rare_dir: Path, tm: int, E: int, tag: str, ref_idx: int) -> Path:
    return _rare_target_dir(rare_dir, tm) / f"ref_flip_idx_E{E}_{tag}_ref{ref_idx}.npy"


def _ref_pred_class_path(rare_dir: Path, tm: int, E: int, tag: str, ref_idx: int) -> Path:
    return _rare_target_dir(rare_dir, tm) / f"ref_pred_class_E{E}_{tag}_ref{ref_idx}.npy"


def _ref_pred_conf_path(rare_dir: Path, tm: int, E: int, tag: str, ref_idx: int) -> Path:
    return _rare_target_dir(rare_dir, tm) / f"ref_pred_confidence_E{E}_{tag}_ref{ref_idx}.npy"


def _ref_ref_ckpt_path(rare_dir: Path, tm: int, tag: str, ref_idx: int) -> Path:
    return _rare_target_dir(rare_dir, tm) / f"ref_ref_model_{tag}_ref{ref_idx}.pth"


def _ref_logits_path(rare_dir: Path, tm: int, E: int, tag: str, ref_idx: int) -> Path:
    return _rare_target_dir(rare_dir, tm) / f"ref_logits_E{E}_{tag}_ref{ref_idx}.npy"


_REF_LOSSES_RE = re.compile(r"^ref_losses_E(\d+)_(.+)_ref(\d+)\.npy$")


def generate_reference_logits(
    experiment_dir: str,
    targets: List[int],
    logit_config: dict,
    num_refs: int = 1,
    logger=None,
    overwrite: bool = False,
    nb_augmentation: int = 2,
) -> None:
    """Cache each reference model's logits so the logit-replacement defense can
    blend on a basis identical to the target/shadow ``logits.npy``.

    The reference per-epoch trace (``ref_losses_*``) is a single-view, bf16,
    train-mode eval and is NOT a valid basis for blending against the attack
    logits. We therefore (re)generate the reference's final-checkpoint logits
    through the SAME ``LogitsGenerator`` that produced the target logits
    (A=``nb_augmentation``, the experiment's ``inference_mode``, fp32), and save
    ``ref_logits_E{E}_{tag}_ref{r}.npy`` (shape ``(N, A, C)``) next to
    ``ref_losses_*``. E and tag are read from the existing ``ref_losses`` files
    so the names align. One forward per reference; no retraining.
    """
    import copy
    from rare.utils.common import load_checkpoint_safe
    from rare.utils.logits import LogitsGenerator

    cfg = copy.deepcopy(logit_config)
    cfg["inference_data_augmentations"] = {
        "mode": "lira_aug", "nb_augmentation": int(nb_augmentation)}
    cfg.setdefault("logits_generation", {})["overwrite"] = True
    gen = LogitsGenerator(cfg, logger)
    exp = Path(experiment_dir)

    for tm in targets:
        td = _rare_target_dir(exp, int(tm))
        for r in range(int(num_refs)):
            match = next(
                (m for p in sorted(td.glob(f"ref_losses_E*_*_ref{r}.npy"))
                 if (m := _REF_LOSSES_RE.match(p.name)) and int(m.group(3)) == r),
                None,
            )
            if match is None:
                raise FileNotFoundError(
                    f"No ref_losses_E*_*_ref{r}.npy for target {tm} in {td}")
            E, tag = int(match.group(1)), match.group(2)
            out = _ref_logits_path(exp, int(tm), E, tag, r)
            if out.exists() and not overwrite:
                if logger:
                    logger.info(f"  [ref logits] cached: {out.name}")
                continue
            ckpt = _ref_ref_ckpt_path(exp, int(tm), tag, r)
            if not ckpt.exists():
                alt = td / f"ref_model_{tag}_ref{r}.pth"  # legacy duplicate name
                if alt.exists():
                    ckpt = alt
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"No reference checkpoint for target {tm} ref {r}: {ckpt}")
            gen._inner_model, _ = load_checkpoint_safe(
                str(ckpt), gen._inner_model, gen.device, logger)
            gen.model = gen._inner_model
            logits = gen._get_model_logits().cpu().numpy().astype(np.float32)
            np.save(out, logits)
            if logger:
                logger.info(f"  [ref logits] saved {out.name} {logits.shape}")


def _loss_and_margin_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample (CE loss, signed margin, pred_class, pred_confidence) from raw logits + true classes.

    Returns 4 tensors (all CPU/GPU-resident, caller .cpu().numpy()s):
      per_example      (B,) fp32
      margin           (B,) fp32  -- z[true] - max_{c != true} z[c]
      pred_class       (B,) int16
      pred_confidence  (B,) fp32  -- softmax(logits)[pred_class]
    """
    per_example = nn.functional.cross_entropy(logits, y, reduction="none")
    logits_f32 = logits.float()
    true_logit = logits_f32.gather(1, y.long().view(-1, 1)).squeeze(1)
    masked = logits_f32.scatter(
        1, y.long().view(-1, 1),
        torch.full_like(logits_f32[:, :1], float("-inf")),
    )
    max_other = masked.max(dim=1).values
    margin = true_logit - max_other
    probs = nn.functional.softmax(logits_f32, dim=1)
    top1_conf, pred_cls = probs.max(dim=1)
    return per_example.float(), margin, pred_cls.to(torch.int16), top1_conf.float()


# =============================================================================
# reference training (model-only, used for cache repair)
# =============================================================================
def train_reference_model_only(
    experiment_dir: str,
    target_model: int,
    config: dict,
    train_dataset_raw,
    membership_mask: np.ndarray,
    num_classes: int,
    device: str,
    n_epochs: int,
    shadow_seed: int,
    shadow_batch_size: int,
    shadow_num_workers: int,
    save_path: Path,
    show_progress: bool = True,
    init_state_path: Optional[Path] = None,
) -> str:
    """Train ONLY the reference model (no per-epoch eval / no margin caching).
    Used to repair missing reference checkpoints when (loss, margin) trajectories are
    already cached.

    When `init_state_path` is provided, the model is initialised from that
    state_dict (the target's saved init weights) so target and reference share starting
    weights.
    """
    mem = membership_mask.astype(bool)
    non_member_idx = np.where(~mem)[0]
    if non_member_idx.size == 0:
        raise RuntimeError("No non-members available to train reference reference model.")

    dev = torch.device(device)
    torch.manual_seed(int(shadow_seed))
    np.random.seed(int(shadow_seed))

    model = get_model(int(num_classes), **config.get("model", {})).to(dev)
    if init_state_path is not None and Path(str(init_state_path)).exists():
        sd = torch.load(str(init_state_path), map_location=dev)
        model.load_state_dict(sd, strict=True)
    train_tf = build_transforms(config, train=True)

    train_subset = TransformSubset(train_dataset_raw, non_member_idx, transform=train_tf)
    train_loader = DataLoader(
        train_subset,
        batch_size=int(shadow_batch_size),
        shuffle=True,
        num_workers=int(shadow_num_workers),
        pin_memory=(dev.type == "cuda"),
        persistent_workers=False,
        drop_last=False,
    )

    _override_cfg = {**config, "training": {**config.get("training", {})}}
    _override_cfg["training"].setdefault("epochs", int(n_epochs))
    optimizer = get_optimizer(model, _override_cfg)
    scheduler, _ = get_scheduler(optimizer, _override_cfg, 1)  # CPU paths step per-epoch
    criterion = nn.CrossEntropyLoss()

    pb = ProgressBar(f"ref_model_only_ref_seed{int(shadow_seed)}", total=int(n_epochs), enabled=show_progress)
    for epoch in range(int(n_epochs)):
        model.train()
        for x, y in train_loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
        scheduler.step()
        pb.update(epoch + 1)
    pb.close()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "shadow_seed": int(shadow_seed),
        "num_classes": int(num_classes),
        "trained_epochs": int(n_epochs),
        "note": "model_only_repair",
    }
    torch.save(payload, save_path)
    return str(save_path)


# =============================================================================
# Top-level orchestrator
# =============================================================================
def load_or_train_multiple_references(
    experiment_dir: str,
    target_model: int,
    config: dict,
    train_dataset_raw,
    membership_mask: np.ndarray,
    num_classes: int,
    device: str,
    n_epochs: int,
    cache_tag: str,
    num_references: int,
    reference_seeds: Optional[List[int]],
    load_if_exists: bool,
    save_cache: bool,
    show_progress: bool,
    flip_nonmember_frac: float,
    flip_nonmember_max: int,
    flip_seed: int,
    eval_batch_size: int,
    shadow_batch_size: int,
    shadow_num_workers: int,
    save_model_checkpoints: bool = True,
    auto_repair_missing_ckpt: bool = True,
    strict_require_all_refs: bool = False,
    rare_models_dir: Optional[str] = None,
    gpu_resident_data: Optional[dict] = None,
    amp_dtype: str = "bf16",
    init_state_path: Optional[str] = None,
    runtime_csv: Optional[str] = None,
    runtime_extras: Optional[dict] = None,
    random_init_after_first: bool = False,
    bootstrap_additional_refs: bool = False,
    bootstrap_frac: float = 0.7,
) -> Tuple[List[np.ndarray], np.ndarray, List[str], List[int]]:
    """Returns (ref_losses_list_NE, flip_idx_master, full_tags_per_ref, reference_seeds_used).

    Margins are written to disk (ref_margin_E*_..._ref{r}.npy) but not returned
    here — downstream scoring loads them via `load_ref_margin_for_target`
    in lockstep with the loss list.

    `gpu_resident_data` (optional): dict with keys
    {"images_uint8", "labels", "mean", "std"} — all on GPU. When provided,
    reference training uses the GPU-resident path. Falls back to CPU DataLoader when None.

    `init_state_path` (optional): path to the target's saved init weights.
    When provided, reference #0 (and, if `random_init_after_first=False`, every
    subsequent reference too) is initialised from this state_dict so target and reference
    share starting weights. The reference residual then isolates the data-flip
    effect rather than init noise; the per-ref `cache_tag` picks up a
    `__sharedinit` suffix.

    `random_init_after_first` (default False): when True, references at index >= 1
    use random init seeded by `reference_seeds[r]` instead of loading
    `init_state_path`. The corresponding per-ref tag picks up a
    `__randinit_seed{S}` suffix so the cache is filesystem-disambiguated
    from the shared-init reference #0. reference #0 itself is unchanged. Used to build a
    diverse reference ensemble for M > 1 (one canonical reference + M-1
    independent shadow models on the same flipped data).

    `runtime_csv` / `runtime_extras` (optional): when both set, each reference
    reference's wall time is appended to the unified runtime.csv as
    phase=ref_retrain, item=target_<tm>_ref_<r>.

    Returns a LIST of full tags (one per ref). For M=1 or for
    random_init_after_first=False, all entries are identical. For
    M>1 with random_init_after_first=True, ref 0's tag carries
    `__sharedinit` and refs 1..M-1 carry `__randinit_seed{S}`.
    """
    rare_dir = Path(rare_models_dir) if rare_models_dir else (Path(experiment_dir) / "rare_models")
    num_refs = max(1, int(num_references))

    if reference_seeds is None:
        reference_seeds_used = list(range(num_refs))
    else:
        reference_seeds_used = list(reference_seeds)[:num_refs]
        while len(reference_seeds_used) < num_refs:
            reference_seeds_used.append(len(reference_seeds_used))

    init_state_path_resolved: Optional[Path] = None
    if init_state_path is not None and Path(str(init_state_path)).exists():
        init_state_path_resolved = Path(str(init_state_path))

    base_tag = (
        f"{str(cache_tag)}__flipf{float(flip_nonmember_frac):.3f}"
        f"__flipm{int(flip_nonmember_max)}__s{int(flip_seed)}"
    )

    # Per-ref init + tag policy. The shared-init reference #0 keeps the historical
    # `__sharedinit` suffix verbatim so the M=1 cache on disk is unchanged.
    # Refs 1..M-1, when `random_init_after_first` is True, get an
    # `__randinit_seed{S}` suffix that's unambiguous on disk.
    init_paths_per_ref: List[Optional[Path]] = []
    full_tags_per_ref: List[str] = []
    for ref_idx in range(num_refs):
        ref_seed = int(reference_seeds_used[ref_idx])
        if ref_idx == 0 or not bool(random_init_after_first):
            init_paths_per_ref.append(init_state_path_resolved)
            suffix = "__sharedinit" if init_state_path_resolved is not None else ""
        else:
            init_paths_per_ref.append(None)
            suffix = f"__randinit_seed{ref_seed}"
        # Additional refs (idx > 0) get a __boot{frac} suffix when bootstrap
        # is enabled so the cache file is filesystem-disambiguated from the
        # full-training-set version of the same ref index.
        if bool(bootstrap_additional_refs) and ref_idx > 0:
            suffix = f"{suffix}__boot{float(bootstrap_frac):.2f}"
        full_tags_per_ref.append(f"{base_tag}{suffix}")

    print(f"\n{'=' * 60}")
    print(f"Reference references: num={num_refs} | CONCAT")
    print(f"Reference seeds: {reference_seeds_used}")
    if random_init_after_first and num_refs > 1:
        print("Init policy: ref0=shared with target; ref1..M-1=random per-seed")
    else:
        print(
            f"Init policy: shared with target (all refs)"
            if init_state_path_resolved is not None
            else "Init policy: random per-seed (all refs)"
        )
    for r in range(num_refs):
        print(f"  ref{r} tag: {full_tags_per_ref[r]}")
    print(f"{'=' * 60}")

    ref_losses_list_NE: List[np.ndarray] = []
    flip_idx_master = None

    # Lazy import so non-reference call paths don't pay the cost.
    if runtime_csv is not None:
        from rare.utils.common import PhaseTimer
    else:
        PhaseTimer = None

    for ref_idx in range(num_refs):
        ref_seed = int(reference_seeds_used[ref_idx])
        full_tag = full_tags_per_ref[ref_idx]
        init_for_this_ref = init_paths_per_ref[ref_idx]
        losses_path = _ref_losses_path(rare_dir, int(target_model), int(n_epochs), full_tag, ref_idx)
        margin_path = _ref_margin_path(rare_dir, int(target_model), int(n_epochs), full_tag, ref_idx)
        flip_path = _ref_flip_path(rare_dir, int(target_model), int(n_epochs), full_tag, ref_idx)
        pred_class_path = _ref_pred_class_path(rare_dir, int(target_model), int(n_epochs), full_tag, ref_idx)
        pred_conf_path = _ref_pred_conf_path(rare_dir, int(target_model), int(n_epochs), full_tag, ref_idx)
        ckpt_p = _ref_ref_ckpt_path(rare_dir, int(target_model), full_tag, ref_idx)

        cache_complete = losses_path.exists() and margin_path.exists()
        if bool(load_if_exists) and cache_complete:
            print(f"\n[Ref {ref_idx}] Loading cached trajectories: {losses_path.name}")
            arr = np.load(losses_path, mmap_mode=None).astype(np.float32, copy=False)
            flip_idx_local = np.array([], dtype=np.int64)
            if flip_path.exists():
                flip_idx_local = np.load(flip_path, mmap_mode=None).astype(np.int64, copy=False)

            if arr.ndim == 2 and arr.shape[1] == int(n_epochs):
                ref_losses_list_NE.append(arr)
                if flip_idx_master is None:
                    flip_idx_master = flip_idx_local
                print(
                    f"  Loaded: losses={arr.shape}, margin_present=True, "
                    f"flip_n={flip_idx_local.size}"
                )

                if save_model_checkpoints and not ckpt_p.exists():
                    msg = f"Missing reference model checkpoint for ref{ref_idx}: {ckpt_p.name}"
                    if strict_require_all_refs:
                        raise FileNotFoundError(msg)
                    print(f"  NOTE: {msg}")
                    if bool(auto_repair_missing_ckpt):
                        print(f"  -> Repair: training reference model ONLY (seed={ref_seed}) to create checkpoint...")
                        saved = train_reference_model_only(
                            experiment_dir=experiment_dir,
                            target_model=target_model,
                            config=config,
                            train_dataset_raw=train_dataset_raw,
                            membership_mask=membership_mask,
                            num_classes=num_classes,
                            device=device,
                            n_epochs=n_epochs,
                            shadow_seed=ref_seed,
                            shadow_batch_size=shadow_batch_size,
                            shadow_num_workers=shadow_num_workers,
                            save_path=ckpt_p,
                            show_progress=show_progress,
                            init_state_path=init_for_this_ref,
                        )
                        print(f"  -> Repaired: saved {Path(saved).name}")
                continue
            else:
                print(f"  Cache invalidated (losses.shape={arr.shape}); retraining.")

        print(f"\n[Ref {ref_idx}] Training reference reference model (seed={ref_seed})...")

        timer_ctx = (
            PhaseTimer(
                runtime_csv, phase="ref_retrain",
                item=f"target_{int(target_model)}_ref_{ref_idx}",
                **(runtime_extras or {}),
            )
            if (PhaseTimer is not None)
            else None
        )
        if timer_ctx is not None:
            timer_ctx.__enter__()
        # Compute per-ref bootstrap subset of target's non-members when
        # bootstrap is enabled for this ref (i.e., ref_idx > 0 with
        # bootstrap_additional_refs=True).  ref 0 stays on the full set.
        train_index_override = None
        if bool(bootstrap_additional_refs) and ref_idx > 0:
            non_member_idx_full = np.where(~membership_mask.astype(bool))[0].astype(np.int64)
            rng_boot = np.random.RandomState(int(ref_seed))
            n_boot = int(round(float(bootstrap_frac) * non_member_idx_full.size))
            train_index_override = rng_boot.choice(
                non_member_idx_full, size=n_boot, replace=True
            ).astype(np.int64)

        try:
            if gpu_resident_data is not None:
                (
                    ref_arr,
                    margin_arr,
                    pred_class_arr,
                    pred_conf_arr,
                    flip_arr,
                    saved_ckpt,
                ) = train_single_reference_reference_gpu(
                    images_uint8_gpu=gpu_resident_data["images_uint8"],
                    labels_gpu=gpu_resident_data["labels"],
                    mean_gpu=gpu_resident_data["mean"],
                    std_gpu=gpu_resident_data["std"],
                    config=config,
                    membership_mask=membership_mask,
                    num_classes=num_classes,
                    device=device,
                    n_epochs=n_epochs,
                    show_progress=show_progress,
                    flip_nonmember_frac=flip_nonmember_frac,
                    flip_nonmember_max=flip_nonmember_max,
                    flip_seed=flip_seed,
                    eval_batch_size=eval_batch_size,
                    shadow_seed=ref_seed,
                    shadow_batch_size=shadow_batch_size,
                    save_model_checkpoint=bool(save_model_checkpoints and save_cache),
                    model_checkpoint_path=ckpt_p,
                    amp_dtype=amp_dtype,
                    init_state_path=init_for_this_ref,
                    train_index_override=train_index_override,
                )
            else:
                (
                    ref_arr,
                    margin_arr,
                    pred_class_arr,
                    pred_conf_arr,
                    flip_arr,
                    saved_ckpt,
                ) = train_single_reference_reference(
                    experiment_dir=experiment_dir,
                    target_model=target_model,
                    config=config,
                    train_dataset_raw=train_dataset_raw,
                    membership_mask=membership_mask,
                    num_classes=num_classes,
                    device=device,
                    n_epochs=n_epochs,
                    show_progress=show_progress,
                    flip_nonmember_frac=flip_nonmember_frac,
                    flip_nonmember_max=flip_nonmember_max,
                    flip_seed=flip_seed,
                    eval_batch_size=eval_batch_size,
                    shadow_seed=ref_seed,
                    shadow_batch_size=shadow_batch_size,
                    shadow_num_workers=shadow_num_workers,
                    save_model_checkpoint=bool(save_model_checkpoints and save_cache),
                    model_checkpoint_path=ckpt_p,
                    init_state_path=init_for_this_ref,
                )
        finally:
            if timer_ctx is not None:
                timer_ctx.__exit__(None, None, None)

        ref_losses_list_NE.append(ref_arr.astype(np.float32, copy=False))
        if flip_idx_master is None:
            flip_idx_master = flip_arr

        if bool(save_cache):
            losses_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(losses_path, ref_arr.astype(np.float32))
            np.save(margin_path, margin_arr.astype(np.float32))
            np.save(flip_path, flip_arr.astype(np.int64))
            np.save(pred_class_path, pred_class_arr.astype(np.int16))
            np.save(pred_conf_path, pred_conf_arr.astype(np.float32))
            print(f"  Saved: {losses_path.name}, {margin_path.name}, "
                  f"{pred_class_path.name}, {pred_conf_path.name}")
            if saved_ckpt is not None:
                print(f"  Saved: {Path(saved_ckpt).name}")

    if flip_idx_master is None:
        flip_idx_master = np.array([], dtype=np.int64)

    return ref_losses_list_NE, flip_idx_master, full_tags_per_ref, reference_seeds_used


# =============================================================================
# reference training (CPU DataLoader path)
# =============================================================================
def train_single_reference_reference(
    experiment_dir: str,
    target_model: int,
    config: dict,
    train_dataset_raw,
    membership_mask: np.ndarray,
    num_classes: int,
    device: str,
    n_epochs: int,
    show_progress: bool,
    flip_nonmember_frac: float,
    flip_nonmember_max: int,
    flip_seed: int,
    eval_batch_size: int,
    shadow_seed: int,
    shadow_batch_size: int,
    shadow_num_workers: int,
    save_model_checkpoint: bool,
    model_checkpoint_path: Path,
    init_state_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    """Returns (reference_losses, reference_margin, flip_idx, saved_ckpt).

    Both per-epoch matrices are shape (N, E) fp32; entries for samples not in
    members ∪ flip stay NaN. `init_state_path` (optional) shares init with the
    target.
    """
    mem = membership_mask.astype(bool)
    N = len(train_dataset_raw)
    non_member_idx = np.where(~mem)[0]
    member_idx = np.where(mem)[0]

    rng = np.random.RandomState(int(flip_seed))
    flip_idx = np.array([], dtype=np.int64)
    if non_member_idx.size > 0 and float(flip_nonmember_frac) > 0.0:
        want = int(round(float(flip_nonmember_frac) * non_member_idx.size))
        want = max(0, min(want, int(flip_nonmember_max), int(non_member_idx.size)))
        if want > 0:
            flip_idx = rng.choice(non_member_idx, size=want, replace=False).astype(np.int64)

    print(f"  Train on non-members: n={int(non_member_idx.size)}")
    print(f"  Track members (reference): n={int(member_idx.size)}")
    print(f"  Track flip non-members: n={int(flip_idx.size)}")

    dev = torch.device(device)
    torch.manual_seed(int(shadow_seed))
    np.random.seed(int(shadow_seed))

    model = get_model(int(num_classes), **config.get("model", {})).to(dev)
    if init_state_path is not None and Path(str(init_state_path)).exists():
        sd = torch.load(str(init_state_path), map_location=dev)
        model.load_state_dict(sd, strict=True)
    train_tf = build_transforms(config, train=True)
    eval_tf = build_transforms(config, train=False)

    train_subset = TransformSubset(train_dataset_raw, non_member_idx, transform=train_tf)
    train_loader = DataLoader(
        train_subset,
        batch_size=int(shadow_batch_size),
        shuffle=True,
        num_workers=int(shadow_num_workers),
        pin_memory=(dev.type == "cuda"),
        persistent_workers=False,
        drop_last=False,
    )

    eval_member_subset = TransformSubset(train_dataset_raw, member_idx, transform=eval_tf)
    eval_member_indexed = IndexedDataset(eval_member_subset)
    eval_member_loader = DataLoader(
        eval_member_indexed,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(shadow_num_workers),
        pin_memory=(dev.type == "cuda"),
        persistent_workers=False,
        drop_last=False,
    )

    if flip_idx.size > 0:
        eval_flip_subset = TransformSubset(train_dataset_raw, flip_idx, transform=eval_tf)
        eval_flip_indexed = IndexedDataset(eval_flip_subset)
        eval_flip_loader = DataLoader(
            eval_flip_indexed,
            batch_size=int(eval_batch_size),
            shuffle=False,
            num_workers=int(shadow_num_workers),
            pin_memory=(dev.type == "cuda"),
            persistent_workers=False,
            drop_last=False,
        )
    else:
        eval_flip_loader = None

    _override_cfg = {**config, "training": {**config.get("training", {})}}
    _override_cfg["training"].setdefault("epochs", int(n_epochs))
    optimizer = get_optimizer(model, _override_cfg)
    scheduler, _ = get_scheduler(optimizer, _override_cfg, 1)  # CPU paths step per-epoch
    criterion = nn.CrossEntropyLoss()

    reference_losses = np.full((N, int(n_epochs)), np.nan, dtype=np.float32)
    reference_margin = np.full((N, int(n_epochs)), np.nan, dtype=np.float32)
    reference_pred_class = np.full((N, int(n_epochs)), -1, dtype=np.int16)
    reference_pred_conf = np.full((N, int(n_epochs)), np.nan, dtype=np.float32)
    pb = ProgressBar("shadow_train", total=int(n_epochs), enabled=show_progress)

    def _eval_into_buffers(loader, mapping_idx, epoch):
        for x, y, idx_local in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = model(x)
            per_example, margin, pred_cls, pred_conf = _loss_and_margin_from_logits(logits, y)
            orig_idx = mapping_idx[idx_local.numpy()]
            reference_losses[orig_idx, epoch] = (
                per_example.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            reference_margin[orig_idx, epoch] = (
                margin.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            reference_pred_class[orig_idx, epoch] = (
                pred_cls.detach().cpu().numpy().astype(np.int16, copy=False)
            )
            reference_pred_conf[orig_idx, epoch] = (
                pred_conf.detach().cpu().numpy().astype(np.float32, copy=False)
            )

    for epoch in range(int(n_epochs)):
        model.train()
        for x, y in train_loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            _eval_into_buffers(eval_member_loader, member_idx, epoch)
            if eval_flip_loader is not None:
                _eval_into_buffers(eval_flip_loader, flip_idx, epoch)

        pb.update(epoch + 1)

    pb.close()

    saved_ckpt = None
    if bool(save_model_checkpoint):
        model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "shadow_seed": int(shadow_seed),
            "num_classes": int(num_classes),
            "trained_epochs": int(n_epochs),
        }
        torch.save(payload, model_checkpoint_path)
        saved_ckpt = str(model_checkpoint_path)

    return reference_losses, reference_margin, reference_pred_class, reference_pred_conf, flip_idx, saved_ckpt


# =============================================================================
# reference training (GPU-resident path)
# =============================================================================
def train_single_reference_reference_gpu(
    *,
    images_uint8_gpu: torch.Tensor,
    labels_gpu: torch.Tensor,
    mean_gpu: torch.Tensor,
    std_gpu: torch.Tensor,
    config: dict,
    membership_mask: np.ndarray,
    num_classes: int,
    device: str,
    n_epochs: int,
    show_progress: bool,
    flip_nonmember_frac: float,
    flip_nonmember_max: int,
    flip_seed: int,
    eval_batch_size: int,
    shadow_seed: int,
    shadow_batch_size: int,
    save_model_checkpoint: bool,
    model_checkpoint_path: Path,
    amp_dtype: str = "bf16",
    init_state_path: Optional[Path] = None,
    train_index_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    """GPU-resident reference training. Returns
    (reference_losses, reference_margin, flip_idx, saved_ckpt).

    Required: `images_uint8_gpu` is the (possibly subsampled) training set as
    uint8 on GPU. `labels_gpu` aligns 1:1 with it. `mean_gpu`, `std_gpu` are
    `(1, 3, 1, 1)` normalisation tensors.
    """
    from rare.utils.gpu_dataset import gpu_eval_augment, gpu_train_augment
    from rare.utils.train import _resolve_amp_dtype

    mem = membership_mask.astype(bool)
    N = int(labels_gpu.size(0))
    if N != mem.size:
        raise ValueError(
            f"membership_mask size {mem.size} != labels size {N} (gpu-resident reference)."
        )
    non_member_idx_full = np.where(~mem)[0].astype(np.int64)
    member_idx = np.where(mem)[0].astype(np.int64)

    # Override training set if caller supplied a custom index array (e.g., a
    # bootstrap of non_member_idx for the per-ref subsampling experiment).
    # The override may contain duplicates (bagging) and may be a subset.
    if train_index_override is not None:
        non_member_idx = np.asarray(train_index_override, dtype=np.int64)
        if not np.all(~mem[non_member_idx]):
            raise ValueError(
                "train_index_override must reference only target's non-members."
            )
        n_unique_train = int(np.unique(non_member_idx).size)
        print(
            f"  [gpu-resident reference] Bootstrap train override: "
            f"n_samples={non_member_idx.size}, n_unique={n_unique_train} "
            f"(of {non_member_idx_full.size} target non-members)"
        )
    else:
        non_member_idx = non_member_idx_full
        n_unique_train = int(non_member_idx.size)

    rng = np.random.RandomState(int(flip_seed))
    flip_idx = np.array([], dtype=np.int64)
    if non_member_idx_full.size > 0 and float(flip_nonmember_frac) > 0.0:
        want = int(round(float(flip_nonmember_frac) * non_member_idx_full.size))
        want = max(0, min(want, int(flip_nonmember_max), int(non_member_idx_full.size)))
        if want > 0:
            flip_idx = rng.choice(non_member_idx_full, size=want, replace=False).astype(np.int64)

    print(f"  [gpu-resident reference] Train on non-members: n={int(non_member_idx.size)} (unique={n_unique_train})")
    print(f"  [gpu-resident reference] Track members (reference): n={int(member_idx.size)}")
    print(f"  [gpu-resident reference] Track flip non-members: n={int(flip_idx.size)}")

    dev = torch.device(device)
    torch.manual_seed(int(shadow_seed))
    np.random.seed(int(shadow_seed))

    model = get_model(int(num_classes), **config.get("model", {})).to(dev)
    if init_state_path is not None and Path(str(init_state_path)).exists():
        sd = torch.load(str(init_state_path), map_location=dev)
        model.load_state_dict(sd, strict=True)

    non_member_idx_gpu = torch.from_numpy(non_member_idx).to(dev)
    member_idx_gpu = torch.from_numpy(member_idx).to(dev)
    flip_idx_gpu = torch.from_numpy(flip_idx).to(dev) if flip_idx.size > 0 else None

    bs = int(shadow_batch_size)
    eval_bs = int(eval_batch_size)
    n_train = int(non_member_idx.size)

    # Build optimizer/scheduler via the SAME helpers train.py uses for shadows
    # and targets, so the reference's training trajectory uses identical
    # Nesterov + per-group weight-decay + cosine-with-optional-warmup as the
    # models it's being compared against. Avoids the silent divergence
    # (nesterov=False, wd on biases, lr-field-name mismatch) the old direct
    # SGD construction caused.
    _override_cfg = {**config, "training": {**config.get("training", {})}}
    _override_cfg["training"].setdefault("epochs", int(n_epochs))
    n_full_batches_for_sched = max(1, int(n_train) // int(bs))
    optimizer = get_optimizer(model, _override_cfg)
    scheduler, step_per_batch = get_scheduler(optimizer, _override_cfg, n_full_batches_for_sched)
    criterion = nn.CrossEntropyLoss()

    amp_torch_dtype = _resolve_amp_dtype(amp_dtype)
    use_bf16 = (amp_torch_dtype == torch.bfloat16)
    scaler = torch.amp.GradScaler('cuda') if (not use_bf16 and dev.type == "cuda") else None

    reference_losses = np.full((N, int(n_epochs)), np.nan, dtype=np.float32)
    reference_margin = np.full((N, int(n_epochs)), np.nan, dtype=np.float32)
    reference_pred_class = np.full((N, int(n_epochs)), -1, dtype=np.int16)
    reference_pred_conf = np.full((N, int(n_epochs)), np.nan, dtype=np.float32)
    pb = ProgressBar("shadow_train_gpu", total=int(n_epochs), enabled=show_progress)

    @torch.no_grad()
    def _eval_into_buffers(idx_to_eval_gpu: torch.Tensor, idx_to_eval_cpu: np.ndarray, epoch: int):
        n = idx_to_eval_gpu.size(0)
        losses_gpu = torch.empty(n, dtype=torch.float32, device=dev)
        margin_gpu = torch.empty(n, dtype=torch.float32, device=dev)
        pred_cls_gpu = torch.empty(n, dtype=torch.int16, device=dev)
        pred_conf_gpu = torch.empty(n, dtype=torch.float32, device=dev)

        for i in range(0, n, eval_bs):
            end = min(i + eval_bs, n)
            chunk = idx_to_eval_gpu[i: end]
            x_uint8 = images_uint8_gpu[chunk]
            y = labels_gpu[chunk]
            x = gpu_eval_augment(x_uint8, mean_gpu, std_gpu)
            with torch.amp.autocast('cuda', dtype=amp_torch_dtype):
                logits = model(x)
            per_example, margin, pred_cls, pred_conf = _loss_and_margin_from_logits(logits, y)
            losses_gpu[i:end] = per_example
            margin_gpu[i:end] = margin
            pred_cls_gpu[i:end] = pred_cls
            pred_conf_gpu[i:end] = pred_conf

        reference_losses[idx_to_eval_cpu, epoch] = losses_gpu.cpu().numpy()
        reference_margin[idx_to_eval_cpu, epoch] = margin_gpu.cpu().numpy()
        reference_pred_class[idx_to_eval_cpu, epoch] = pred_cls_gpu.cpu().numpy()
        reference_pred_conf[idx_to_eval_cpu, epoch] = pred_conf_gpu.cpu().numpy()

    # Same GPU augmentation as the target (random crop + flip + normalize).

    n_full_batches = n_train // bs  # drop_last
    for epoch in range(int(n_epochs)):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        train_idx_shuffled = non_member_idx_gpu[perm]
        for i in range(n_full_batches):
            chunk = train_idx_shuffled[i * bs: (i + 1) * bs]
            x_uint8 = images_uint8_gpu[chunk]
            y = labels_gpu[chunk]
            x, y_train = gpu_train_augment(x_uint8, mean_gpu, std_gpu, padding=4, y=y)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_torch_dtype):
                outputs = model(x)
                loss = criterion(outputs, y_train)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None and step_per_batch:
                scheduler.step()
        if scheduler is not None and not step_per_batch:
            scheduler.step()

        model.eval()
        _eval_into_buffers(member_idx_gpu, member_idx, epoch)
        if flip_idx_gpu is not None:
            _eval_into_buffers(flip_idx_gpu, flip_idx, epoch)

        pb.update(epoch + 1)
    pb.close()

    saved_ckpt = None
    if bool(save_model_checkpoint):
        model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "shadow_seed": int(shadow_seed),
            "num_classes": int(num_classes),
            "trained_epochs": int(n_epochs),
        }
        torch.save(payload, model_checkpoint_path)
        saved_ckpt = str(model_checkpoint_path)

    return reference_losses, reference_margin, reference_pred_class, reference_pred_conf, flip_idx, saved_ckpt

