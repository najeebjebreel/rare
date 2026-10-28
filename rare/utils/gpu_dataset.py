"""
GPU-resident dataset + on-GPU augmentation for CIFAR-shaped inputs.

Why: Blackwell + WRN-28-2 leaves the GPU starving on the standard CPU
DataLoader path because random-crop + flip + normalize per image are CPU-side.
By keeping the entire training set as a single GPU tensor and doing
augmentation with vectorised GPU ops, the per-epoch time on RARE retrain
drops from ~3s (CPU-bound) to ~0.5-1s (compute-bound) on CIFAR-10/WRN-28-2.

The matching CPU augmentation pipeline this replaces (see rare/utils/data.py
build_transforms for CIFAR-{10,100} with the default config):
    - RandomCrop(32, padding=4, padding_mode='reflect')
    - RandomHorizontalFlip(p=0.5)
    - ToTensor()                      (uint8 [0,255] -> float [0,1])
    - Normalize(mean=DATASET_STATS[name][0], std=DATASET_STATS[name][1])
For eval: ToTensor + Normalize only.

Per-image augmentation randomness (different (h_off, w_off, flip_mask) per
sample within a batch) matches the CPU pipeline. The exact random numbers
won't match the CPU torchvision RNG, so retrained-target weights will differ
slightly from the original — same caveat as the existing CPU retrain.
"""

import io
import os
import pickle
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100


# Per-dataset normalization stats. Mirrors rare.utils.data.DATASET_STATS
# for the supported subset; the CINIC-10 row matches LT-IQR's published
# transforms for apples-to-apples shadow-pool comparison.
_DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cinic10": ((0.47889522, 0.47227842, 0.43047404),
                (0.24205776, 0.23828046, 0.25874835)),
}

# Datasets we currently support GPU-resident batching for.
SUPPORTED_DATASETS = frozenset({"cifar10", "cifar100", "cinic10"})


def is_gpu_resident_supported(dataset_name: str) -> bool:
    return str(dataset_name).lower() in SUPPORTED_DATASETS


# =============================================================================
# Loading
# =============================================================================
def load_image_dataset_to_gpu(
    dataset_name: str,
    data_dir: str,
    train: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load CIFAR-{10,100} entirely onto GPU as uint8 tensors.

    Returns (images_uint8 (N,3,32,32) on `device`, labels long (N,) on `device`,
    mean (1,3,1,1) fp32 on `device`, std (1,3,1,1) fp32 on `device`).
    """
    name = str(dataset_name).lower()
    if name == "cifar10":
        ds = CIFAR10(root=data_dir, train=bool(train), download=False, transform=None)
        images_np = np.transpose(ds.data, (0, 3, 1, 2)).copy()  # (N,3,32,32) uint8
        labels_np = np.asarray(ds.targets, dtype=np.int64)
    elif name == "cifar100":
        ds = CIFAR100(root=data_dir, train=bool(train), download=False, transform=None)
        images_np = np.transpose(ds.data, (0, 3, 1, 2)).copy()
        labels_np = np.asarray(ds.targets, dtype=np.int64)
    elif name == "cinic10":
        # LMDB-backed: lazy-decode every PNG once, stack to (N,3,32,32) uint8.
        # The build_lmdb_cinic.py script writes pickled (img_bytes, label) pairs
        # under integer-string keys plus a "__len__" sentinel.
        partition = "train" if bool(train) else "test"
        lmdb_path = os.path.join(data_dir, "cinic10", f"{partition}.lmdb")
        images_np, labels_np = _decode_cinic10_lmdb(lmdb_path)
    else:
        raise ValueError(
            f"GPU-resident loader supports {sorted(SUPPORTED_DATASETS)}; got {dataset_name!r}"
        )

    images_gpu = torch.from_numpy(images_np).to(device)
    labels_gpu = torch.from_numpy(labels_np).to(device)

    mean_t, std_t = _DATASET_STATS[name]
    mean_gpu = torch.tensor(mean_t, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_gpu = torch.tensor(std_t, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return images_gpu, labels_gpu, mean_gpu, std_gpu


def _decode_cinic10_lmdb(lmdb_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read every record of a CINIC-10 LMDB into (images_uint8 NCHW, labels)."""
    import lmdb  # imported lazily so non-CINIC paths don't pay the cost
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        subdir=False,
    )
    with env.begin(write=False) as txn:
        raw_len = txn.get(b"__len__")
        if raw_len is None:
            env.close()
            raise KeyError(f"LMDB missing __len__ key: {lmdb_path}")
        n = int(pickle.loads(raw_len))
        images = np.empty((n, 3, 32, 32), dtype=np.uint8)
        labels = np.empty((n,), dtype=np.int64)
        for i in range(n):
            buf = txn.get(str(i).encode())
            if buf is None:
                env.close()
                raise KeyError(f"LMDB missing key {i}: {lmdb_path}")
            img_bytes, label = pickle.loads(buf)
            img = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB"),
                             dtype=np.uint8)  # (32,32,3)
            images[i] = np.transpose(img, (2, 0, 1))
            labels[i] = int(label)
    env.close()
    return images, labels


# =============================================================================
# Per-batch GPU augmentation
# =============================================================================
def gpu_train_augment(
    x_uint8: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    padding: int = 4,
    pad_mode: str = "reflect",
    *,
    y: Optional[torch.Tensor] = None,
):
    """RandomCrop(padding, pad_mode) -> RandomHorizontalFlip(p=0.5) -> Normalize, on GPU.

    `x_uint8`: (B, 3, H, W) uint8. Returns the augmented batch, or `(x, y)`
    when `y` is given (the loader path always passes `y`).
    """
    B, C, H, W = x_uint8.shape
    x = x_uint8.float() / 255.0                                                     # (B,C,H,W)
    if str(pad_mode).lower() == "constant":
        x_pad = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0.0)
    else:
        x_pad = F.pad(x, (padding, padding, padding, padding), mode="reflect")      # (B,C,H+2p,W+2p)

    # All possible (2p+1)x(2p+1) crops as a stride view, then pick one per image.
    crops = x_pad.unfold(2, H, 1).unfold(3, W, 1)                                   # (B,C,2p+1,2p+1,H,W)
    h_off = torch.randint(0, 2 * padding + 1, (B,), device=x.device)
    w_off = torch.randint(0, 2 * padding + 1, (B,), device=x.device)
    out = crops[torch.arange(B, device=x.device), :, h_off, w_off]                  # (B,C,H,W)

    flip_mask = torch.rand(B, device=x.device) < 0.5
    out = torch.where(flip_mask.view(B, 1, 1, 1), out.flip(-1), out)
    out = (out - mean) / std
    return out if y is None else (out, y)


def gpu_eval_augment(
    x_uint8: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Eval-only normalisation (no augmentation). uint8 -> fp32, normalised."""
    x = x_uint8.float() / 255.0
    return (x - mean) / std


# =============================================================================
# DataLoader-compatible wrapper
# =============================================================================
class _DummyDataset:
    """Stub so `loader.dataset.__len__()` works for code that probes it."""

    def __init__(self, n: int):
        self.n = int(n)

    def __len__(self) -> int:
        return self.n


class GPUResidentLoader:
    """Drop-in DataLoader replacement that yields GPU-resident batches.

    Compatible with `train_epoch`, `evaluate`, `compute_topk_logits_and_losses`,
    etc. — they all iterate `(inputs, targets)` and call `inputs.to(device)`,
    which is a no-op when inputs are already on GPU.

    Parameters
    ----------
    images_uint8 : (N_full, C, H, W) uint8 tensor on `device`. Master image bank.
    labels       : (N_full,) long tensor on `device`. Master label bank.
    indices      : (M,) long tensor on `device`. Subset of `[0, N_full)` to iterate.
    batch_size   : int.
    mean, std    : (1, C, 1, 1) fp32 tensors on `device`.
    train        : when True, applies `gpu_train_augment`; else `gpu_eval_augment`.
    shuffle      : reshuffle `indices` each epoch (only meaningful when train=True).
    drop_last    : drop the trailing partial batch.
    crop_padding : padding for RandomCrop (CIFAR default = 4).
    """

    def __init__(
        self,
        images_uint8: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        train: bool = True,
        shuffle: bool = True,
        drop_last: bool = True,
        crop_padding: int = 4,
        pad_mode: str = "reflect",
    ):
        if images_uint8.device != labels.device:
            raise ValueError(
                f"images and labels must be on the same device "
                f"({images_uint8.device} vs {labels.device})"
            )
        self._images = images_uint8
        self._labels = labels
        self._indices = torch.as_tensor(indices, dtype=torch.long, device=images_uint8.device)
        self._batch_size = int(batch_size)
        self._mean = mean
        self._std = std
        self._train = bool(train)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        self._crop_padding = int(crop_padding)
        self._pad_mode = str(pad_mode)
        self.dataset = _DummyDataset(self._indices.size(0))

    def __len__(self) -> int:
        n = self._indices.size(0)
        if self._drop_last:
            return n // self._batch_size
        return (n + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = self._indices.size(0)
        if self._shuffle:
            perm = torch.randperm(n, device=self._indices.device)
            order = self._indices[perm]
        else:
            order = self._indices

        for i in range(0, n, self._batch_size):
            idx_chunk = order[i: i + self._batch_size]
            if self._drop_last and idx_chunk.size(0) < self._batch_size:
                continue
            x_uint8 = self._images[idx_chunk]
            y = self._labels[idx_chunk]
            if self._train:
                x, y_out = gpu_train_augment(
                    x_uint8, self._mean, self._std,
                    padding=self._crop_padding, pad_mode=self._pad_mode,
                    y=y,
                )
                yield x, y_out
            else:
                x = gpu_eval_augment(x_uint8, self._mean, self._std)
                yield x, y


# =============================================================================
# Loader-set builder (matches rare.utils.data.create_data_loaders' return shape)
# =============================================================================
def create_gpu_resident_loaders(
    *,
    train_images_uint8: torch.Tensor,
    train_labels: torch.Tensor,
    test_images_uint8: torch.Tensor,
    test_labels: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    member_indices: np.ndarray,
    train_cfg: dict,
    device: torch.device,
    crop_padding: int = 4,
    pad_mode: str = "reflect",
) -> Tuple[GPUResidentLoader, GPUResidentLoader, GPUResidentLoader, GPUResidentLoader]:
    """Build the four loaders the rest of the pipeline expects.

    Returns (train_loader, test_loader, train_eval_loader, per_epoch_eval_loader).
    """
    tr_cfg = train_cfg.get("training", {})
    bs = int(tr_cfg.get("batch_size", 256))
    eval_bs = int(tr_cfg.get("eval_batch_size", bs * 4))

    n_train_full = train_images_uint8.size(0)
    all_train_idx = torch.arange(n_train_full, device=device, dtype=torch.long)
    n_test = test_images_uint8.size(0)
    all_test_idx = torch.arange(n_test, device=device, dtype=torch.long)
    member_idx_gpu = torch.as_tensor(member_indices, device=device, dtype=torch.long)

    train_loader = GPUResidentLoader(
        train_images_uint8, train_labels, member_idx_gpu,
        batch_size=bs, mean=mean, std=std,
        train=True, shuffle=True, drop_last=True,
        crop_padding=crop_padding, pad_mode=pad_mode,
    )
    test_loader = GPUResidentLoader(
        test_images_uint8, test_labels, all_test_idx,
        batch_size=eval_bs, mean=mean, std=std,
        train=False, shuffle=False, drop_last=False,
    )
    train_eval_loader = GPUResidentLoader(
        train_images_uint8, train_labels, member_idx_gpu,
        batch_size=eval_bs, mean=mean, std=std,
        train=False, shuffle=False, drop_last=False,
    )
    per_epoch_eval_loader = GPUResidentLoader(
        train_images_uint8, train_labels, all_train_idx,
        batch_size=eval_bs, mean=mean, std=std,
        train=False, shuffle=False, drop_last=False,
    )
    return train_loader, test_loader, train_eval_loader, per_epoch_eval_loader
