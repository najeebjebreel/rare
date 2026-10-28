"""Generate and persist logits for trained shadow models."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rare.utils.data import load_dataset, resolve_subsample
from rare.utils.model import get_model
from rare.utils.common import load_checkpoint_safe


class LogitsGenerator:
    """Generate and save logits for shadow models with deterministic augmentations."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.experiment_dir = config.get('experiment', {}).get('checkpoint_dir', 'experiments')
        logits_cfg = config.get('logits_generation', {})

        target_fallback = config.get('target_model', 'best')
        self.target_model = logits_cfg.get('target_model', target_fallback)
        self.overwrite_logits = logits_cfg.get('overwrite', False)

        # Inference mode: 'eval' (standard PyTorch; Pollock LT-IQR; our default
        # for new benchmarks) or 'train' (Carlini Objax-style; legacy, used
        # for CIFAR-10/CIFAR-100 to stay consistent with the cached shadow
        # logits there). Config keys checked in order:
        #   logits_generation.inference_mode  > top-level inference_mode
        self.inference_mode = str(
            logits_cfg.get('inference_mode',
                           config.get('inference_mode', 'eval'))
        ).lower()
        # bf16 autocast for the forward pass. Logits are cast back to fp32
        # immediately after the model call, so downstream attack-side numerics
        # see fp32. Default off to preserve legacy numerics; enable for
        # ~2x speedup on Blackwell tensor cores.
        self.bf16 = bool(logits_cfg.get('bf16', config.get('bf16', False)))

        aug_cfg = logits_cfg.get('inference_data_augmentations',
                                 config.get('inference_data_augmentations', {}))
        self.aug_type = aug_cfg.get('mode', 'none')
        self.spatial_shift = 0
        self.spatial_stride = 1
        self.use_horizontal_flip = False
        if self.aug_type == 'lira_aug':
            nb_augmentation = int(aug_cfg.get('nb_augmentation', 18))
            if nb_augmentation == 0:
                pass
            elif nb_augmentation == 2:
                self.use_horizontal_flip = True
            elif nb_augmentation == 18:
                self.spatial_shift = 1
                self.use_horizontal_flip = True
            elif nb_augmentation == 50:
                self.spatial_shift = 2
                self.use_horizontal_flip = True
            else:
                raise ValueError("Unsupported LiRA nb_augmentation value (use 0, 2, 18, or 50)")

        use_cuda = config.get('use_cuda', True)
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        self.logger.info("Loading dataset for logits generation...")
        self.full_dataset, _, _ = load_dataset(config, mode='inference')

        # Apply the experiment's saved deterministic subsample (if any) so
        # generated logits align with keep_indices over the (possibly subsampled)
        # train pool used at training time.
        ds_cfg_local = config.get('dataset', {})
        n_data_points = ds_cfg_local.get('n_data_points', None)
        subsample_seed = int(ds_cfg_local.get('subsample_seed', config.get('seed', 42)))
        n_full_pool = len(self.full_dataset)
        sub_idx = resolve_subsample(
            n_full_pool, n_data_points, subsample_seed,
            save_path=os.path.join(self.experiment_dir, 'subsample_indices.npy'),
        )
        if sub_idx.size != n_full_pool:
            self.full_dataset = Subset(self.full_dataset, sub_idx.tolist())
            self.logger.info(
                f"[subsample] applying {sub_idx.size}/{n_full_pool} via subsample_indices.npy"
            )

        train_cfg = config.get('training', {})
        batch_size = logits_cfg.get('batch_size', config.get('batch_size', train_cfg.get('batch_size', 2048)))
        num_workers = logits_cfg.get('num_workers', config.get('num_workers', train_cfg.get('num_workers', 0)))
        self.batch_size = int(batch_size)
        self.data_loader = DataLoader(
            self.full_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        model_cfg = config.get('model', {})
        num_classes = config.get('dataset', {}).get('num_classes', 10)
        # `_inner_model` is the raw nn.Module. `self.model` is what the
        # forward pass uses -- either the same eager module, or a compiled
        # wrapper around it. State_dict loads always go through
        # `_inner_model` so OptimizedModule's `_orig_mod.` key prefix never
        # leaks into the checkpoint mismatch path.
        self._inner_model = get_model(num_classes, **model_cfg).to(self.device)
        self.model = self._inner_model

        # torch.compile (PyTorch 2.x). fp32-preserving graph compilation +
        # optional CUDA-graph replay. First batch triggers compilation
        # (~10-30 s); subsequent shadows reuse the cached graph since
        # `load_state_dict()` mutates the underlying tensors in-place and
        # the compiled graph holds them by reference.
        self.torch_compile = bool(logits_cfg.get('torch_compile',
                                                 config.get('torch_compile', False)))
        self.compile_mode = str(logits_cfg.get('compile_mode',
                                               config.get('compile_mode', 'reduce-overhead')))
        if self.torch_compile:
            try:
                self.logger.info(f"Enabling torch.compile (mode={self.compile_mode!r})")
                self.model = torch.compile(self._inner_model, mode=self.compile_mode)
            except Exception as e:
                self.logger.warning(f"torch.compile unavailable / failed ({e!r}); using eager mode")
                self.model = self._inner_model

        self.num_shadow_models = train_cfg.get('num_shadow_models', 1)
        self.logger.info(f"Configured {self.num_shadow_models} shadow models for logits generation")

        # GPU-resident materialization of the full inference tensor. One-time
        # cost; eliminates per-batch CPU->GPU transfer + DataLoader IPC overhead
        # for the 266-shadow inference sweep. ~600 MB on disk for CIFAR-sized
        # data; trivial on a 96 GB GPU.
        self.logger.info("Pre-loading inference dataset to GPU...")
        load_loader = DataLoader(
            self.full_dataset,
            batch_size=1024,
            shuffle=False,
            num_workers=num_workers,
        )
        chunks = []
        for images, _ in load_loader:
            chunks.append(images.to(self.device, non_blocking=True))
        self.images_gpu = torch.cat(chunks, dim=0).contiguous()
        del chunks
        bytes_mb = self.images_gpu.element_size() * self.images_gpu.numel() / 1024 / 1024
        self.logger.info(
            f"GPU-resident inference tensor: shape={tuple(self.images_gpu.shape)} "
            f"dtype={self.images_gpu.dtype} ({bytes_mb:.1f} MB)"
        )

    def _get_model_logits(self, data_loader=None):
        """Run inference under the configured `inference_mode` and augmentation
        grid. Returns logits of shape (N, A, C) where A = num_flips * num_shifts.

        Optimization (2026-05-13):
          - Uses the GPU-resident `self.images_gpu` tensor materialized in
            __init__ -- no per-batch CPU->GPU transfer or DataLoader workers.
          - Folds all A augmentations of a B-image batch into a single
            forward pass over (B*A, C, H, W). Replaces the previous per-aug
            for-loop. ~5-10x wall-clock speedup on a Blackwell-class GPU.

        Semantic note for `inference_mode='train'`: BN uses current-batch
        stats; folded augmentation changes the batch composition (B*A
        correlated views instead of B distinct samples per forward pass),
        so the resulting logits differ from the legacy per-aug path. The
        difference is internally self-consistent across the 266-shadow
        sweep (every shadow sees the same batching), so LiRA / RMIA /
        Attack-R signals remain valid -- just on a new numerical baseline.
        """
        # Inference mode:
        #   'eval' -- standard PyTorch: BN uses running stats, dropout
        #     disabled. Matches Pollock et al. (USENIX'25 LT-IQR).
        #   'train' -- Carlini 2022 Objax-style: BN uses current-batch stats,
        #     dropout disabled per-module. Used for CIFAR-10 / CIFAR-100
        #     legacy benchmarks.
        if self.inference_mode == 'train':
            self.model.train()
            for module in self.model.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                    module.eval()
        elif self.inference_mode == 'eval':
            self.model.eval()
        else:
            raise ValueError(
                f"Unknown inference_mode={self.inference_mode!r}; use 'eval' or 'train'"
            )

        reflect = bool(self.use_horizontal_flip)
        shift = int(self.spatial_shift)
        stride = int(self.spatial_stride)

        images_gpu = self.images_gpu                                    # (N, C, H, W)
        N, C_img, H, W = images_gpu.shape
        bs = int(self.batch_size)

        out_chunks = []
        with torch.no_grad():
            for start in tqdm(range(0, N, bs), desc="Processing batches"):
                batch = images_gpu[start:start + bs]                     # (B, C, H, W) on GPU
                B = batch.shape[0]

                if shift > 0:
                    padded = F.pad(batch, [shift, shift, shift, shift], mode='reflect')
                else:
                    padded = batch

                # LiRA augmentation order: flip outer, dx middle, dy inner.
                # Materialise each (B, C, H, W) crop, then stack along a new
                # A axis -> (B, A, C, H, W) -> reshape (B*A, C, H, W) for
                # one fused forward pass.
                crops = []
                flip_variants = [padded]
                if reflect:
                    flip_variants.append(torch.flip(padded, dims=[3]))
                for flipped in flip_variants:
                    for dx in range(0, 2 * shift + 1, stride):
                        for dy in range(0, 2 * shift + 1, stride):
                            crops.append(flipped[:, :, dy:dy + H, dx:dx + W])

                aug = torch.stack(crops, dim=1)                          # (B, A, C, H, W)
                A = aug.shape[1]
                aug = aug.reshape(B * A, C_img, H, W).contiguous()

                if self.bf16:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = self.model(aug)                         # bf16 forward
                    logits = logits.float()                              # cast back to fp32
                else:
                    logits = self.model(aug)                             # fp32 forward
                logits = logits.view(B, A, -1)
                out_chunks.append(logits)

        result = torch.cat(out_chunks, dim=0)                            # (N, A, C_out)
        self.logger.debug(f"Generated logits with {result.shape[1]} augmentations per sample")
        return result

    def generate_logits(self):
        """Generate and save logits for every shadow model."""
        if self.target_model == 'best':
            target_model_prefix = 'best_model.pth'
        else:
            target_model_prefix = f"checkpoint_epoch{self.target_model}.pth"

        # Fast-fail: verify every shadow that will actually be re-scored this
        # run has the requested checkpoint BEFORE spending ~1 h on logit
        # regeneration. Skipping a shadow mid-loop would leave attacks.py
        # with a hole in its (M, N, A) signal tensor.
        #
        # When `overwrite=false`, shadows whose `logits.npy` already exists
        # on disk are NOT re-scored, so we don't need their checkpoint and a
        # missing one is harmless. When `overwrite=true`, every shadow gets
        # re-scored, so every checkpoint must be present.
        self.logger.info(
            f"Checkpoint policy: target_model={self.target_model!r} -> '{target_model_prefix}', "
            f"inference_mode={self.inference_mode!r}, overwrite={self.overwrite_logits}"
        )
        missing = []
        for i in range(self.num_shadow_models):
            ckpt_p = os.path.join(self.experiment_dir, f'model_{i}', target_model_prefix)
            if os.path.exists(ckpt_p):
                continue
            logits_p = os.path.join(self.experiment_dir, f'model_{i}', 'logits', 'logits.npy')
            if (not self.overwrite_logits) and os.path.exists(logits_p):
                # Will be skipped anyway because cached logits already exist.
                continue
            missing.append(i)
        if missing:
            preview = missing[:10]
            raise FileNotFoundError(
                f"{len(missing)} shadow(s) are missing '{target_model_prefix}' "
                f"and would need to be (re)scored. First missing: model_{preview}. "
                f"Either retrain those shadows, copy in a checkpoint, or change "
                f"`target_model` in the experiment's logit_config.yaml (NOT recommended)."
            )

        # Augmentation descriptor logged once at INFO; per-model details (load,
        # inference, save) demoted to DEBUG so the 266-iteration sweep doesn't
        # spam the console. Outer-loop progress shown via tqdm.
        aug_desc = f"{self.aug_type}"
        if self.spatial_shift > 0:
            aug_desc += f" (shift={self.spatial_shift}, stride={self.spatial_stride})"
        if self.use_horizontal_flip:
            aug_desc += " + horizontal_flip"
        self.logger.info(f"Running inference over {self.num_shadow_models} shadows | augmentations: {aug_desc}")

        for i in tqdm(range(self.num_shadow_models), desc="Shadows", unit="model"):
            self.logger.debug(f"Inference using shadow model {i}...")
            ckpt = os.path.join(self.experiment_dir, f'model_{i}', target_model_prefix)

            logits_dir = os.path.join(self.experiment_dir, f'model_{i}', 'logits')
            os.makedirs(logits_dir, exist_ok=True)
            logits_path = os.path.join(logits_dir, 'logits.npy')
            if os.path.exists(logits_path) and not self.overwrite_logits:
                self.logger.debug(f"Logits already exist for model {i}, skipping")
                continue

            try:
                self.logger.debug(f"Loading checkpoint from {ckpt}")
                # Load into the un-compiled inner module. When torch.compile
                # wraps it, `self.model` is the OptimizedModule whose params
                # alias `_inner_model`'s -- so the forward sees fresh weights.
                self._inner_model, _ = load_checkpoint_safe(ckpt, self._inner_model, self.device, self.logger)

                logits = self._get_model_logits()
                logits = logits.cpu().unsqueeze(1).numpy()  # [N, 1, A, C]
                self.logger.debug(f"Logits shape: {logits.shape}")
                np.save(logits_path, logits)
                self.logger.debug(f"Saved logits to {logits_path}")

            except Exception as exc:
                self.logger.error(f"Error on model {i}: {exc}")

        self.logger.info(f"Completed logits generation for {self.num_shadow_models} shadows")
