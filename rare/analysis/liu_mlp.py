"""
Liu et al. (TRAJECTORYMIA, KDD 2022) MLP head adapted to the white-box defender.

Liu's published method (Eq. 8, Sec. 3.4):
  - Distill target on auxiliary data, keep N intermediate distilled models.
  - Token = sample's loss across the N+1 distilled models (single source).
  - Train a 4-layer MLP on shadow trajectories with member / non-member labels.
  - At inference, feed the target's distilled trajectory through the MLP.

Our adaptation (defender setting, no distillation):
  - Use the actual per-epoch training loss trace (already on disk for every
    target model in this project) as the trajectory token.
  - Train MLP cross-target leave-one-out: 9 of 10 targets supply labelled
    trajectories, evaluate on the held-out target.
  - Token shape and supervised MLP head match Liu; only the trajectory source
    is changed (training axis vs distillation axis).

This file provides:
  - LiuMLP module
  - data assembly helpers
  - train_liu_mlp / predict_membership_proba
  - score_liu_mlp end-to-end pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LiuMLP(nn.Module):
    """4-layer fully-connected classifier (Liu et al. Sec. 3.4, 4 layers).

    Defaults are the config we report for LT-MLP: hidden widths 256-128-64
    with dropout 0.2, adapted from the released ``MLP_BLACKBOX`` (which used
    512-128-32, no dropout) for the single-reference auditing regime -- the
    released widths/no-dropout with plain SGD underfit or diverge here (see
    the appendix). Widths/dropout are overridable to reproduce the release
    (``hidden_dims=(512, 128, 32), dropout=0.0``).
    """

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (256, 128, 64), dropout: float = 0.2):
        super().__init__()
        dims = [int(input_dim), *(int(h) for h in hidden_dims)]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True), nn.Dropout(p=float(dropout))]
        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_robust_scaler(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(X, axis=0).astype(np.float32, copy=False)
    q75 = np.nanquantile(X, 0.75, axis=0)
    q25 = np.nanquantile(X, 0.25, axis=0)
    iqr = np.maximum(q75 - q25, eps).astype(np.float32, copy=False)
    return med, iqr


def _apply_robust_scaler(X: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    Xn = (X - med) / iqr
    return np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def load_trajectory_token(experiment_dir: str, target_model: int, n_data_points: Optional[int] = None) -> np.ndarray:
    """
    Load per-epoch training losses for `target_model` from the target retrain
    (`experiment_dir/model_<t>/per_epoch_losses.npy`) and return shape (N, T).
    Source file is (T, N); we transpose so each row is one sample's trajectory.

    Reads from `$EXP/model_<t>/` so Liu-MLP scores from the SAME retrained
    target trajectory as RARE and the same model LiRA / RMIA / Attack-R
    attacked — apples-to-apples baselines per EXPERIMENTS.md §8.
    """
    path = Path(experiment_dir) / f"model_{int(target_model)}" / "per_epoch_losses.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"per_epoch_losses.npy missing under model_<t>/ for target {target_model}: {path}\n"
            f"Run scripts/train.py --config {experiment_dir}/train_config.yaml "
            f"--trace-mode loss_only --targets {target_model}"
        )
    pel = np.load(path).astype(np.float32, copy=False)
    if pel.ndim != 2:
        raise ValueError(f"per_epoch_losses must be 2D, got shape {pel.shape}")
    pel = pel.T
    if n_data_points is not None:
        pel = pel[: int(n_data_points)]
    return pel


def load_membership(experiment_dir: str, target_model: int, n_data_points: Optional[int] = None) -> np.ndarray:
    keep = np.load(Path(experiment_dir) / "keep_indices.npy")
    mem = keep[int(target_model)].astype(np.int64)
    if n_data_points is not None:
        mem = mem[: int(n_data_points)]
    return mem


def assemble_cross_target_dataset(
    experiment_dir: str,
    train_targets: List[int],
    n_data_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack trajectories and labels from `train_targets`. Returns (X, y)."""
    Xs, ys = [], []
    for t in train_targets:
        Xs.append(load_trajectory_token(experiment_dir, t, n_data_points=n_data_points))
        ys.append(load_membership(experiment_dir, t, n_data_points=n_data_points))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def train_liu_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: str,
    hidden_dims: Tuple[int, ...] = (256, 128, 64),
    dropout: float = 0.2,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    optimizer: str = "adamw",
    amp: bool = True,
    seed: int = 0,
    log_every: int = 5,
) -> nn.Module:
    """Train a Liu-style MLP on labelled trajectories. Returns the model in eval mode.

    ``optimizer="adamw"`` (default, reported config): AdamW + cosine annealing.
    ``optimizer="sgd"``: plain SGD with momentum and a fixed learning rate, the
    authors' released MIA.py recipe (use with the released 512-128-32/no-dropout
    widths to reproduce the faithful port). Either way the last-epoch model is
    returned (no best-checkpoint selection), matching the release.
    """
    dev = torch.device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    X_t = torch.from_numpy(X_train.astype(np.float32, copy=False))
    y_t = torch.from_numpy(y_train.astype(np.int64, copy=False))
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
    )

    model = LiuMLP(input_dim=X_train.shape[1], hidden_dims=tuple(hidden_dims), dropout=float(dropout)).to(dev)
    opt_name = str(optimizer).lower()
    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs))
    elif opt_name == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay)
        )
        sched = None
    else:
        raise ValueError(f"optimizer must be 'adamw' or 'sgd', got {optimizer!r}")

    use_amp = bool(amp and dev.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    print(f"[liu_mlp] training: N={X_train.shape[0]}, T={X_train.shape[1]}, epochs={epochs}, bs={batch_size}, lr={lr}")
    model.train()
    for ep in range(1, int(epochs) + 1):
        loss_sum, n_seen = 0.0, 0
        for bx, by in loader:
            bx = bx.to(dev, non_blocking=True)
            by = by.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(bx)
                loss = nn.functional.cross_entropy(logits, by)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            loss_sum += float(loss.item()) * by.numel()
            n_seen += int(by.numel())
        if sched is not None:
            sched.step()
        if ep == 1 or ep % int(log_every) == 0 or ep == int(epochs):
            print(
                f"  [liu_mlp] epoch {ep:02d}/{int(epochs)} | "
                f"loss={loss_sum / max(n_seen, 1):.4f} | lr={opt.param_groups[0]['lr']:.2e}"
            )

    model.eval()
    return model


@torch.no_grad()
def predict_membership_proba(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    """P(member | trajectory) for each row of X, shape (N,)."""
    dev = torch.device(device)
    model.eval()
    out = np.empty(X.shape[0], dtype=np.float32)
    for i in range(0, X.shape[0], int(batch_size)):
        bx = torch.from_numpy(X[i: i + int(batch_size)].astype(np.float32, copy=False)).to(dev, non_blocking=True)
        probs = torch.softmax(model(bx), dim=-1)
        out[i: i + int(batch_size)] = probs[:, 1].detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def score_liu_mlp(
    experiment_dir: str,
    test_target: int,
    target_models: List[int],
    device: str,
    n_data_points: Optional[int] = None,
    hidden_dims: Tuple[int, ...] = (256, 128, 64),
    dropout: float = 0.2,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    optimizer: str = "adamw",
    standardize: bool = True,
    amp: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """
    Cross-target leave-one-out scoring: train on the OTHER targets' trajectories,
    return per-sample membership probability on `test_target`. Shape (N,).

    Defaults are the reported LT-MLP config: AdamW(lr=1e-3, wd=5e-4)+cosine,
    batch 1024, 20 epochs, MLP 256/128/64 with dropout 0.2, and a train-only
    robust (median/IQR) scaler on the trajectories. To reproduce the authors'
    released recipe (which underfits/diverges here), pass
    ``optimizer="sgd", hidden_dims=(512, 128, 32), dropout=0.0, epochs=100,
    batch_size=128, lr=1e-2, weight_decay=1e-4, standardize=False``.
    """
    train_targets = [int(t) for t in target_models if int(t) != int(test_target)]
    if not train_targets:
        raise ValueError("Need at least one other target for training")

    X_train, y_train = assemble_cross_target_dataset(experiment_dir, train_targets, n_data_points=n_data_points)
    X_test = load_trajectory_token(experiment_dir, int(test_target), n_data_points=n_data_points)

    if standardize:
        med, iqr = _fit_robust_scaler(X_train)
        X_train = _apply_robust_scaler(X_train, med, iqr)
        X_test = _apply_robust_scaler(X_test, med, iqr)
    else:
        # Release feeds raw losses; guard against non-finite entries only.
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    model = train_liu_mlp(
        X_train, y_train, device=device,
        hidden_dims=hidden_dims, dropout=dropout, epochs=epochs,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, momentum=momentum,
        optimizer=optimizer, amp=amp, seed=seed,
    )
    return predict_membership_proba(model, X_test, device=device, batch_size=int(batch_size) * 2)
