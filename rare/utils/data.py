"""
Dataset loading and transform utilities.

Fast path for CINIC-10: LMDB-backed dataset with lazy env opening and a
per-get transaction (no persistent txn, no buffers=True), which empirically
gives the best random-access performance for shadow-model sampling.
"""

import os
import io
import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, GTSRB
from PIL import Image


DATASET_STATS = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'cifar100': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'cinic10': ((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
    'gtsrb': ((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}

DATASET_NATIVE_SIZE = {
    "cifar10": 32,
    "cifar100": 32,
    "cinic10": 32,
    "gtsrb": None,
    "imagenet": None
}


class TabularDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.transform = transform
        self.features = features if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32)
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class LMDBCINIC10(Dataset):

    def __init__(self, lmdb_path: str, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None  # lazy per worker

        with lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            subdir=False,
        ) as env:
            with env.begin(write=False) as txn:
                raw_len = txn.get(b"__len__")
                if raw_len is None:
                    raise KeyError(f"LMDB missing __len__ key: {lmdb_path}")
                self.length = int(pickle.loads(raw_len))

        self.targets = []
        with lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            subdir=False,
        ) as env:
            with env.begin(write=False) as txn:
                for i in range(self.length):
                    buf = txn.get(str(i).encode())
                    if buf is None:
                        raise KeyError(f"LMDB missing key {i}")
                    _, label = pickle.loads(buf)
                    self.targets.append(label)

    def _ensure_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                subdir=False,
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_env()
        with self.env.begin(write=False) as txn:
            buf = txn.get(str(int(idx)).encode())
            if buf is None:
                raise KeyError(f"LMDB missing key {idx}")
            img_bytes, label = pickle.loads(buf)

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


class TransformSubset(Dataset):
    def __init__(self, full_data, keep_indices, transform=None):
        if isinstance(keep_indices, np.ndarray) and keep_indices.dtype == bool:
            indices = np.where(keep_indices)[0]
        else:
            indices = keep_indices
        self.subset = Subset(full_data, indices)
        self.transform = transform
        self._len = len(self.subset)

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self._len


def build_transforms(config, train=True):
    ds_cfg = config.get('dataset', {})
    name = ds_cfg.get('name', '').lower()

    if name in ['purchase', 'texas', 'location']:
        return None

    input_size = ds_cfg.get('input_size', 32)
    aug_list = config.get('train_data_augmentation', []) if train else []
    ops = []

    native_input_size = DATASET_NATIVE_SIZE.get(name, None)
    if native_input_size is None:
        ops.append(transforms.Resize((input_size, input_size)))
    else:
        if input_size != native_input_size:
            ops.append(transforms.Resize((input_size, input_size)))

    if train:
        for aug in aug_list:
            if aug == 'random_crop':
                pad = max(1, input_size // 8)
                ops.append(transforms.RandomCrop(input_size, padding=pad, padding_mode='reflect'))
            elif aug == 'random_flip':
                ops.append(transforms.RandomHorizontalFlip())
            elif aug == 'random_rotation':
                angle = 10 if name == 'gtsrb' else 15
                ops.append(transforms.RandomRotation(angle))
            elif aug == 'color_jitter':
                if name == 'gtsrb':
                    ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
                else:
                    ops.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))

    ops.append(transforms.ToTensor())

    if train and 'cutout' in aug_list:
        cutout_size = input_size // 2
        scale_ratio = (cutout_size * cutout_size) / float(input_size * input_size)
        ops.append(transforms.RandomErasing(
            p=1.0, scale=(scale_ratio, scale_ratio), ratio=(1.0, 1.0), value=0
        ))

    if ('normalize' in aug_list) or (not train):
        pretrained = config.get('model', {}).get('pretrained', False)
        stats_key = 'imagenet' if (pretrained or name == 'imagenet') else name
        mean, std = DATASET_STATS.get(stats_key, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        ops.append(transforms.Normalize(mean, std))

    return transforms.Compose(ops)


def resolve_subsample(
    n_full: int,
    n_target,
    seed: int,
    save_path=None,
):
    """Return sorted long indices into [0, n_full) defining a deterministic subsample.

    Used to match the LT-IQR / RMIA CINIC-10 50k-of-90k setup, but generic for any
    dataset. If `n_target` is None or >= n_full, the full pool is returned. If
    `save_path` is provided and the file exists, it is loaded; otherwise the
    indices are generated from `seed` and persisted there.

    Returns a 1-D `np.int64` array of size `min(n_full, n_target)`.
    """
    if n_target is None or int(n_target) >= int(n_full):
        return np.arange(int(n_full), dtype=np.int64)
    if save_path is not None and os.path.exists(save_path):
        idx = np.load(save_path)
        return idx.astype(np.int64, copy=False)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(int(n_full), size=int(n_target), replace=False).astype(np.int64)
    idx.sort()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        np.save(save_path, idx)
    return idx


def get_keep_indices(dataset_size, num_shadow_models, pkeep, seed=0):
    np.random.seed(seed)
    if num_shadow_models <= 0:
        return np.random.uniform(0, 1, size=dataset_size) <= pkeep

    keep_scores = np.random.uniform(0, 1, size=(num_shadow_models, dataset_size))
    threshold = int(pkeep * num_shadow_models)
    order = np.argsort(keep_scores, axis=0)
    keep_indices = order < threshold
    return keep_indices


def load_dataset(config, mode='training'):
    dataset_name = config['dataset']['name'].lower()
    data_dir = config.get('dataset', {}).get('data_dir', 'data')

    if mode == 'training':
        train_transform = build_transforms(config, train=True)
        test_transform = build_transforms(config, train=False)
    else:
        train_transform = test_transform = build_transforms(config, train=False)

    if dataset_name == 'cifar10':
        train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
        train_dataset_eval = CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)

    elif dataset_name == 'cifar100':
        train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
        train_dataset_eval = CIFAR100(root=data_dir, train=True, download=True, transform=test_transform)

    elif dataset_name == 'cinic10':
        train_dataset = LMDBCINIC10(os.path.join(data_dir, "cinic10", "train.lmdb"), transform=train_transform)
        test_dataset = LMDBCINIC10(os.path.join(data_dir, "cinic10", "test.lmdb"), transform=test_transform)
        train_dataset_eval = LMDBCINIC10(os.path.join(data_dir, "cinic10", "train.lmdb"), transform=test_transform)

    elif dataset_name == 'gtsrb':
        train_dataset = GTSRB(root=data_dir, split='train', download=True, transform=train_transform)
        test_dataset = GTSRB(root=data_dir, split='test', download=True, transform=test_transform)
        train_dataset_eval = GTSRB(root=data_dir, split='train', download=True, transform=test_transform)

    elif dataset_name == 'purchase':
        data_path = os.path.join(data_dir, dataset_name, 'features_labels.npy')
        full_data = np.load(data_path)
        full_features = torch.tensor(full_data[:, 1:].astype(np.float32))
        full_labels_raw = full_data[:, 0].astype(np.int32) - 1
        full_dataset = TabularDataset(full_features, full_labels_raw, transform=None)
        train_size = int(0.3 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(config.get('seed', 0))
        )
        train_dataset_eval = train_dataset

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if mode == 'training':
        keep_indices = get_keep_indices(
            len(train_dataset),
            config['training']['num_shadow_models'],
            config['dataset']['pkeep'],
            config['seed']
        )
    else:
        keep_indices = None

    return train_dataset, keep_indices, test_dataset, train_dataset_eval


def create_data_loaders(train_dataset, test_dataset, config,
                        shadow_train_eval_dataset=None, per_epoch_eval_dataset=None):
    t_cfg = config.get('training', {})
    batch_size = int(t_cfg.get('batch_size', 128))
    workers = int(t_cfg.get('num_workers', 0))

    pin_memory = bool(config.get('use_cuda', True)) and torch.cuda.is_available()

    loader_kwargs = {
        'num_workers': workers,
        'pin_memory': pin_memory,
        'persistent_workers': workers > 0,
    }
    if workers > 0:
        # 2 is often optimal for LMDB-backed datasets.
        loader_kwargs['prefetch_factor'] = int(t_cfg.get('prefetch_factor', 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )

    eval_batch_size = int(t_cfg.get('eval_batch_size', batch_size * 4))

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs
    )

    train_eval_loader = None
    if shadow_train_eval_dataset is not None:
        train_eval_loader = DataLoader(
            shadow_train_eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            **loader_kwargs
        )

    per_epoch_eval_loader = None
    if per_epoch_eval_dataset is not None:
        per_epoch_eval_loader = DataLoader(
            per_epoch_eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            **loader_kwargs
        )

    return train_loader, test_loader, train_eval_loader, per_epoch_eval_loader
