"""Training loop, optimizers, schedulers, and loss tracking."""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LinearLR, SequentialLR

from rare.utils.common import evaluate


# Mixed-precision dtype dispatch. bf16 is preferred on sm_120 (Blackwell): same
# fp32 exponent range so no GradScaler needed, ~10-20% throughput vs fp16.
_AMP_DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _resolve_amp_dtype(name):
    if name is None:
        return torch.float16
    key = str(name).lower()
    if key not in _AMP_DTYPE_MAP:
        raise ValueError(f"Unsupported amp_dtype {name!r}. Use one of {list(_AMP_DTYPE_MAP)}.")
    return _AMP_DTYPE_MAP[key]


def log_msg(msg, logger=None, level="info"):
    if logger is None:
        print(msg)
    else:
        getattr(logger, level, logger.info)(msg)


def get_optimizer(model, config):
    """Build an optimizer (SGD / Adam / AdamW) from the training config."""
    tr_cfg = config.get('training', {})
    optimizer_name = tr_cfg.get('optimizer', 'sgd').lower()
    lr = float(tr_cfg.get('learning_rate', 0.1))
    weight_decay = float(tr_cfg.get('weight_decay', 5e-4))

    if optimizer_name == 'adamw':
        betas = tr_cfg.get('betas', [0.9, 0.999])
        eps = float(tr_cfg.get('eps', 1e-8))

        no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm.weight', 'bn']
        param_groups = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': weight_decay,
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
            },
        ]

        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )

    elif optimizer_name == 'adam':
        betas = tr_cfg.get('betas', [0.9, 0.999])
        eps = float(tr_cfg.get('eps', 1e-8))

        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    elif optimizer_name == 'sgd':
        momentum = float(tr_cfg.get('momentum', 0.9))

        param_groups = [
            {
                'params': [
                    p for n, p in model.named_parameters() if 'bias' not in n
                ],
                'weight_decay': weight_decay,
            },
            {
                'params': [
                    p for n, p in model.named_parameters() if 'bias' in n
                ],
                'weight_decay': 0.0,
            },
        ]

        optimizer = optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer


def get_scheduler(optimizer, config, steps_per_epoch):
    """Build a LR scheduler. Returns (scheduler, step_per_batch)."""
    tr_cfg = config.get('training', {})
    scheduler_type = tr_cfg.get('lr_scheduler', 'cosine').lower()
    num_epochs = int(tr_cfg.get('epochs', 10))
    warmup_epochs = float(tr_cfg.get('warmup_epochs', 0.0))

    if scheduler_type == 'cosine':
        if warmup_epochs > 0:
            warmup_steps = int(warmup_epochs * steps_per_epoch)
            total_steps = num_epochs * steps_per_epoch

            if warmup_steps <= 0 or warmup_steps >= total_steps:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=num_epochs,
                    eta_min=1e-6,
                )
                return scheduler, False

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )

            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=1e-6,
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
            return scheduler, True
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=1e-6,
            )
            return scheduler, False

    elif scheduler_type == 'onecycle':
        max_lr = float(tr_cfg.get('learning_rate', 0.1)) * 3.0
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.3,
            anneal_strategy='cos',
        )
        return scheduler, True

    else:
        return None, False


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    num_epochs,
    scheduler=None,
    step_per_batch=False,
    log_interval=10,
    use_amp=True,
    logger=None,
    amp_dtype="fp16",
):
    """Run one training epoch (with optional per-batch scheduler step)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    amp_torch_dtype = _resolve_amp_dtype(amp_dtype)
    use_scaler = use_amp and amp_torch_dtype == torch.float16
    scaler = GradScaler('cuda') if use_scaler else None

    num_batches = len(train_loader)
    pbar = tqdm(
        enumerate(train_loader),
        total=num_batches,
        desc=f"Epoch {epoch + 1:3d}/{num_epochs}",
        unit="batch",
        disable=logger is not None,
    )

    for batch_idx, (inputs, targets) in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast('cuda', dtype=amp_torch_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if scheduler and step_per_batch:
            scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        # CutMix yields soft labels (B, num_classes); fall back to argmax of
        # the soft target for the running-accuracy display. CE itself already
        # accepts soft targets natively (PyTorch >= 1.10).
        target_class = targets if targets.dim() == 1 else targets.argmax(dim=1)
        total += target_class.size(0)
        correct += predicted.eq(target_class).sum().item()

        if logger is None and (
            (batch_idx + 1) % log_interval == 0
            or (batch_idx + 1) == num_batches
        ):
            pbar.set_postfix(
                {
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100.0 * correct / total if total > 0 else 0.0,
                }
            )

    avg_loss = running_loss / num_batches
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    test_loader,
    train_eval_loader,
    per_epoch_eval_loader,
    config,
    device,
    save_dir=None,
    logger=None,
    track_losses=False,
    track_margin=False,
    amp_dtype="fp16",
    overwrite_checkpoint=True,
):
    """
    Train a model end-to-end with optimizer/scheduler from config.

    When `track_losses=True`, also records per-epoch per-sample CE losses on
    `per_epoch_eval_loader` and writes them to `<save_dir>/per_epoch_losses.npy`.

    When `track_margin=True`, also records per-epoch per-sample signed logit
    margin = z[true_class] - max_{c != true_class} z[c]. Writes
    `<save_dir>/per_epoch_margin.npy` shape (E, N_eval) fp32. Together with the
    loss matrix this is the 2-channel (loss, margin) trajectory RARE consumes.

    When `overwrite_checkpoint=False` AND `<save_dir>/best_model.pth` already
    exists, the post-loop checkpoint save is skipped. Used by trace-mode
    retrains so the original target weights (and the logits computed against
    them) remain valid; the new trace artefacts are still written.
    """
    tr_cfg = config.get('training', {})
    num_epochs = tr_cfg.get('epochs', 100)
    save_step = tr_cfg.get('save_step', 20)
    save_models = tr_cfg.get('save_models', False)
    early_stop = tr_cfg.get('early_stopping', False)
    patience = tr_cfg.get('patience', 10)
    resume = tr_cfg.get('resume', False)
    use_amp = tr_cfg.get('use_amp', torch.cuda.is_available())
    # When False, skip the per-epoch test eval (and best-model tracking + early
    # stopping that depend on it). The final-epoch weights are saved as
    # best_model.pth and the post-loop evaluate() block produces the standard
    # train/test acc once. Useful for shadow training where per-shadow accuracy
    # spread is small and the per-epoch test eval is pure overhead.
    eval_per_epoch = bool(tr_cfg.get('eval_per_epoch', True))

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(model, config)
    scheduler, step_per_batch = get_scheduler(
        optimizer,
        config,
        len(train_loader),
    )

    best_acc = 0.0
    epochs_no_improve = 0
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    start_epoch = 0
    best_epoch = None

    all_train_losses = None
    if track_losses:
        log_msg("Tracking per-sample losses during training.", logger)
        num_eval_samples = len(per_epoch_eval_loader.dataset)
        all_train_losses = np.zeros(
            (num_epochs, num_eval_samples),
            dtype=np.float32,
        )

        if save_dir is not None and resume:
            losses_path = save_dir / "per_epoch_losses.npy"
            if losses_path.exists():
                try:
                    prev = np.load(losses_path)
                    if prev.shape == (num_epochs, num_eval_samples):
                        all_train_losses[...] = prev.astype(np.float32)
                        log_msg(
                            f"[Resume] Loaded existing per-epoch loss matrix "
                            f"from {losses_path} with shape {prev.shape}",
                            logger,
                        )
                    else:
                        log_msg(
                            f"[Resume] Existing loss matrix at {losses_path} has "
                            f"shape {prev.shape}, expected "
                            f"({num_epochs}, {num_eval_samples}). Reinitializing.",
                            logger,
                        )
                except Exception as e:
                    log_msg(
                        f"[Resume] Failed to load existing loss matrix from "
                        f"{losses_path}: {e}. Reinitializing.",
                        logger,
                    )

    all_train_margin = None
    if track_margin:
        log_msg("Tracking per-sample (loss, margin) during training.", logger)
        num_eval_samples = len(per_epoch_eval_loader.dataset)
        all_train_margin = np.zeros((num_epochs, num_eval_samples), dtype=np.float32)

        if save_dir is not None and resume:
            margin_path = save_dir / "per_epoch_margin.npy"
            if margin_path.exists():
                try:
                    prev = np.load(margin_path)
                    if prev.shape == (num_epochs, num_eval_samples):
                        all_train_margin[...] = prev.astype(np.float32)
                        log_msg(f"[Resume] Loaded margin matrix from {margin_path}", logger)
                except Exception as e:
                    log_msg(f"[Resume] Failed to load margin matrix: {e}. Reinitializing.", logger)

    if resume and save_dir:
        ckpt_paths = list(save_dir.glob('checkpoint_epoch*.pth'))
        if ckpt_paths:
            def _get_epoch(path):
                m = re.search(r'checkpoint_epoch(\d+)\.pth$', str(path))
                return int(m.group(1)) if m else -1

            latest_path = max(ckpt_paths, key=_get_epoch)
            checkpoint = torch.load(latest_path, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

            if start_epoch >= num_epochs:
                log_msg(
                    f"[Resume] Already trained for {num_epochs} epochs, skipping.",
                    logger,
                )
                return model

            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler and 'scheduler' in checkpoint:
                if checkpoint['scheduler'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler'])
            best_acc = checkpoint.get('best_acc', best_acc)
            train_losses = checkpoint.get('train_losses', [])
            test_losses = checkpoint.get('test_losses', [])
            train_accuracies = checkpoint.get('train_accuracies', [])
            test_accuracies = checkpoint.get('test_accuracies', [])
            log_msg(
                f"[Resume] Resumed from {latest_path} at epoch {start_epoch}",
                logger,
            )

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            num_epochs,
            scheduler=scheduler,
            step_per_batch=step_per_batch,
            use_amp=use_amp,
            logger=logger,
            amp_dtype=amp_dtype,
        )

        is_last_epoch = (epoch + 1 == num_epochs)
        # Always evaluate on the last epoch, regardless of `eval_per_epoch`, so
        # the canonical Epoch line for the final model carries the eval-time
        # (no-aug) train + test metrics that downstream code reports as model
        # utility. For all other epochs, only eval when explicitly requested.
        if eval_per_epoch or is_last_epoch:
            test_loss, test_acc = evaluate(
                model,
                test_loader,
                criterion,
                device,
                use_amp=use_amp,
                logger=logger,
                epoch=epoch,
                num_epochs=num_epochs,
            )
        else:
            test_loss = float('nan')
            test_acc = float('nan')

        # Eval-time train metrics for the last epoch (members only, no aug).
        eval_train_loss = float('nan')
        eval_train_acc = float('nan')
        if is_last_epoch:
            eval_train_loss, eval_train_acc = evaluate(
                model, train_eval_loader, criterion, device, use_amp=use_amp,
            )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if track_losses or track_margin:
            if track_margin:
                epoch_losses, epoch_margins = compute_loss_and_margin(
                    model,
                    per_epoch_eval_loader,
                    device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                if all_train_margin is not None:
                    n_eval = epoch_margins.shape[0]
                    if all_train_margin.shape[1] != n_eval:
                        raise ValueError(
                            "Mismatch in eval samples: "
                            f"all_train_margin.shape[1]={all_train_margin.shape[1]}, "
                            f"epoch_margins.shape[0]={n_eval}"
                        )
                    all_train_margin[epoch, :] = epoch_margins.astype(np.float32)
            else:
                epoch_losses, _ = compute_loss_values(
                    model,
                    per_epoch_eval_loader,
                    device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
            if all_train_losses is not None:
                if all_train_losses.shape[1] != epoch_losses.shape[0]:
                    raise ValueError(
                        "Mismatch in eval samples: "
                        f"all_train_losses.shape[1]={all_train_losses.shape[1]}, "
                        f"epoch_losses.shape[0]={epoch_losses.shape[0]}"
                    )
                all_train_losses[epoch, :] = epoch_losses.astype(np.float32)

        if scheduler and not step_per_batch:
            scheduler.step()

        if save_dir:
            if save_models and (epoch + 1) % save_step == 0:
                ckpt_file = save_dir / f'checkpoint_epoch{epoch + 1}.pth'
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler else None,
                        'best_acc': best_acc,
                        'train_losses': train_losses,
                        'test_losses': test_losses,
                        'train_accuracies': train_accuracies,
                        'test_accuracies': test_accuracies,
                    },
                    ckpt_file,
                )
                log_msg(f"[Save ] Saved checkpoint: {ckpt_file}", logger)

            if eval_per_epoch:
                if test_acc > best_acc:
                    best_acc = test_acc
                    epochs_no_improve = 0
                    best_file = save_dir / 'best_model.pth'
                    torch.save(
                        {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_acc': best_acc,
                            'train_losses': train_losses,
                            'test_losses': test_losses,
                            'train_accuracies': train_accuracies,
                            'test_accuracies': test_accuracies,
                        },
                        best_file,
                    )
                    log_msg(
                        f"[Save ] Best model @ epoch {epoch + 1:3d} with acc {test_acc:.2f}%",
                        logger,
                    )
                else:
                    epochs_no_improve += 1

        if eval_per_epoch and early_stop and epochs_no_improve >= patience:
            log_msg(
                f"[EarlyStop] Stopped after {epoch + 1} epochs "
                f"(no improvement in {patience})",
                logger,
            )
            break

        if is_last_epoch:
            # Canonical final-epoch line: train-aug stats + eval-time (no-aug)
            # train + test stats for the final model.
            log_msg(
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Train(aug) Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Train(eval) Loss: {eval_train_loss:.4f} Acc: {eval_train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
                f"LR: {optimizer.param_groups[0]['lr']:.5f}",
                logger,
            )
        elif eval_per_epoch:
            log_msg(
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
                f"LR: {optimizer.param_groups[0]['lr']:.5f}",
                logger,
            )
        else:
            log_msg(
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"LR: {optimizer.param_groups[0]['lr']:.5f}",
                logger,
            )

    # When per-epoch eval is off, no best-model was tracked: persist the
    # final-epoch weights as best_model.pth so the post-loop block + downstream
    # pipeline have a model to load. Trace-mode retrains pass
    # `overwrite_checkpoint=False` so an existing best_model.pth (and the
    # logits computed against it) stay intact; only the trace .npy files get
    # (re)written.
    if save_dir and not eval_per_epoch:
        best_file = save_dir / 'best_model.pth'
        if (not overwrite_checkpoint) and best_file.exists():
            log_msg(
                f"[Save ] Preserving existing best_model.pth (overwrite_checkpoint=False)",
                logger,
            )
        else:
            torch.save(
                {
                    'epoch': num_epochs,
                    'state_dict': model.state_dict(),
                    'best_acc': float('nan'),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accuracies': train_accuracies,
                    'test_accuracies': test_accuracies,
                },
                best_file,
            )
            log_msg(
                f"[Save ] Final-epoch model saved as best_model.pth (eval_per_epoch=False)",
                logger,
            )

    best_model_file = save_dir / 'best_model.pth' if save_dir else None
    final_train_loss = final_train_acc = None
    final_test_loss = final_test_acc = None

    # Post-loop final eval: load the saved checkpoint (best epoch if
    # eval_per_epoch=True; final epoch if eval_per_epoch=False) and run
    # one train+test pass to populate metrics.csv. Cheap (one forward
    # over train_eval + test loaders) and gives the user the final
    # model's accuracy without per-epoch overhead.
    #
    # Skip the redundant re-eval + duplicate log line when the loop ran
    # to the last epoch (which now does the same eval inline as part of
    # the canonical Epoch line) — just reuse the already-computed values
    # for metrics.csv.
    last_epoch_eval_done = (
        not np.isnan(eval_train_loss) and not np.isnan(test_loss)
    )
    if last_epoch_eval_done:
        final_train_loss, final_train_acc = float(eval_train_loss), float(eval_train_acc)
        final_test_loss, final_test_acc = float(test_loss), float(test_acc)
        best_epoch = num_epochs
    elif best_model_file and best_model_file.exists():
        best_ckpt = torch.load(best_model_file, map_location=device)
        model.load_state_dict(best_ckpt['state_dict'])
        best_epoch = best_ckpt.get('epoch', best_epoch)

        final_train_loss, final_train_acc = evaluate(
            model,
            train_eval_loader,
            criterion,
            device,
            use_amp=use_amp,
        )
        final_test_loss, final_test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
            use_amp=use_amp,
        )
        which = "Final-epoch" if not eval_per_epoch else "Best"
        log_msg(
            f"[Eval] {which} model from {best_model_file} "
            f"\n| Train Loss: {final_train_loss:.4f}, Acc: {final_train_acc:.2f}% | "
            f"Test Loss: {final_test_loss:.4f}, Acc: {final_test_acc:.2f}%",
            logger,
        )
    elif train_losses and test_losses:
        final_train_loss = train_losses[-1]
        final_test_loss = test_losses[-1]
        final_train_acc = train_accuracies[-1]
        final_test_acc = test_accuracies[-1]
        log_msg(
            "[Warning] No best_model.pth found, using last epoch metrics "
            "for CSV summary.",
            logger,
        )

    if save_dir and final_train_loss is not None:
        csv_path = save_dir / "metrics.csv"
        with csv_path.open("w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(
                [
                    "Best_epoch",
                    "Train_loss",
                    "Test_loss",
                    "Train_acc (%)",
                    "Test_acc (%)",
                ]
            )
            writer_csv.writerow(
                [
                    best_epoch if best_epoch is not None else "N/A",
                    f"{final_train_loss:.4f}",
                    f"{final_test_loss:.4f}",
                    f"{final_train_acc:.2f}",
                    f"{final_test_acc:.2f}",
                ]
            )
        log_msg(f"Wrote metrics to {csv_path}", logger)

    if track_losses and save_dir and all_train_losses is not None:
        losses_path = save_dir / "per_epoch_losses.npy"
        np.save(losses_path, all_train_losses)
        log_msg(
            f"Saved per-epoch per-sample train losses to {losses_path}",
            logger,
        )
        log_msg(
            f"Shape of saved loss matrix: {all_train_losses.shape}",
            logger,
        )

    if track_margin and save_dir and all_train_margin is not None:
        margin_path = save_dir / "per_epoch_margin.npy"
        np.save(margin_path, all_train_margin)
        log_msg(
            f"Saved per-epoch per-sample logit margin to {margin_path} "
            f"(shape={all_train_margin.shape}, dtype={all_train_margin.dtype})",
            logger,
        )

    return model


@torch.no_grad()
def get_model_predictions(model, data_loader, device, use_amp=True):
    """Get predictions, targets, and per-class probabilities from a loader."""
    model.eval()
    preds, targets, confs = [], [], []

    for inputs, labs in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)
        if use_amp:
            with autocast('cuda'):
                out = model(inputs)
        else:
            out = model(inputs)
        prob = F.softmax(out, dim=1)
        _, p = prob.max(1)
        preds.append(p.cpu().numpy())
        targets.append(labs.cpu().numpy())
        confs.append(prob.cpu().numpy())

    return (
        np.concatenate(preds),
        np.concatenate(targets),
        np.concatenate(confs),
    )


@torch.no_grad()
def compute_loss_values(model, data_loader, device, use_amp=True, amp_dtype="fp16"):
    """Compute per-sample CE loss values across a loader."""
    model.eval()
    losses, targets = [], []
    amp_torch_dtype = _resolve_amp_dtype(amp_dtype)

    for inputs, labs in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)
        if use_amp:
            with autocast('cuda', dtype=amp_torch_dtype):
                out = model(inputs)
                per_example = F.cross_entropy(out, labs, reduction='none')
        else:
            out = model(inputs)
            per_example = F.cross_entropy(out, labs, reduction='none')

        losses.append(per_example.cpu().numpy())
        targets.append(labs.cpu().numpy())

    return np.concatenate(losses), np.concatenate(targets)


@torch.no_grad()
def compute_loss_and_margin(model, data_loader, device, use_amp=True, amp_dtype="fp16"):
    """Single forward pass; returns 2 per-sample channels.

    Channels:
      losses   (N,) fp32   -- per-sample CE loss
      margins  (N,) fp32   -- z[true] - max_{c != true} z[c]

    This is the (loss, margin) per-epoch trace RARE consumes for the IN/OUT
    trajectory comparison.
    """
    model.eval()
    loss_chunks, margin_chunks = [], []
    amp_torch_dtype = _resolve_amp_dtype(amp_dtype)

    for inputs, labs in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)
        if use_amp:
            with autocast('cuda', dtype=amp_torch_dtype):
                out = model(inputs)
                per_example = F.cross_entropy(out, labs, reduction='none')
        else:
            out = model(inputs)
            per_example = F.cross_entropy(out, labs, reduction='none')

        out_f32 = out.float()                                                      # (B, C)
        true_logit = out_f32.gather(1, labs.long().view(-1, 1)).squeeze(1)         # (B,)
        masked = out_f32.scatter(
            1, labs.long().view(-1, 1),
            torch.full_like(out_f32[:, :1], float("-inf")),
        )
        max_other = masked.max(dim=1).values                                       # (B,)
        margin = (true_logit - max_other)                                          # (B,)

        loss_chunks.append(per_example.float().cpu().numpy())
        margin_chunks.append(margin.cpu().numpy().astype(np.float32, copy=False))

    return (
        np.concatenate(loss_chunks, axis=0),
        np.concatenate(margin_chunks, axis=0),
    )
