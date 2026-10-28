"""Logging, seeding, config, and checkpoint utilities."""

import os
import time
import yaml
import logging
import numpy as np
import torch
import random
from torch.amp import autocast
import csv
import math
from pathlib import Path


def append_runtime(csv_path, phase: str, item: str, seconds: float, **extra) -> None:
    """Append a single runtime row to `csv_path`. Creates the file with a header
    on first write. `extra` columns (e.g. dataset, arch, n_data_points, loading)
    are persisted alongside so cross-experiment comparison stays self-describing.

    Schema: timestamp, phase, item, seconds, plus any extra kwargs.
    """
    csv_path = str(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True) if os.path.dirname(csv_path) else None
    base_fields = ["timestamp", "phase", "item", "seconds"]
    extra_fields = sorted(extra.keys())
    fieldnames = base_fields + extra_fields
    write_header = not os.path.exists(csv_path)
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": str(phase),
        "item": str(item),
        "seconds": f"{float(seconds):.3f}",
    }
    for k in extra_fields:
        row[k] = str(extra[k])
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


class PhaseTimer:
    """Context-manager that records elapsed wall time to a runtime.csv.

    Usage:
        with PhaseTimer(csv_path, phase="train", item=f"shadow_{i}", dataset="cifar10"):
            train_one_shadow(...)
    """

    def __init__(self, csv_path, phase: str, item: str, **extra):
        self.csv_path = csv_path
        self.phase = phase
        self.item = item
        self.extra = extra
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._t0
        # Don't suppress exceptions, but still log timing on failure.
        try:
            append_runtime(self.csv_path, self.phase, self.item, elapsed, **self.extra)
        except Exception:
            pass
        return False

def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger with file and console handlers.

    The default level is INFO. Per-iteration verbose logs across the project
    are written at DEBUG, so the default console output is concise. Set the
    env var `RARE_LOG_LEVEL=DEBUG` (or pass `level=logging.DEBUG`) to see
    the verbose per-shadow / per-target output, useful when something fails
    mid-sweep and you need to know exactly which iteration broke.
    """
    env_level = os.environ.get("RARE_LOG_LEVEL", "").upper()
    if env_level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        level = getattr(logging, env_level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_seed(seed):
    """Set random seed for Python, NumPy, and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Trade strict reproducibility for cudnn autotuning speed.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def save_config(config, save_path):
    """Save configuration dict to a YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def parse_overrides(override_list):
    """Parse a list of `key.path=value` strings into a nested dict."""
    overrides = {}
    for item in override_list:
        if '=' not in item:
            raise ValueError(f"Invalid override format: {item}. Use key.subkey=value format.")
        key, value = item.split('=', 1)

        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

        keys = key.split('.')
        d = overrides
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return overrides


def recursive_update(d, u):
    """Recursively update nested dictionary `d` with values from `u`."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            recursive_update(d[k], v)
        elif v in [None, 'none'] and d.get(k) not in [None, 'none']:
            continue
        else:
            d[k] = v


def cleanup_gpu_memory():
    """Clear CUDA cache and run garbage collection."""
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_checkpoint_safe(checkpoint_path, model, device=None, logger=None):
    """Load a checkpoint, handling both `state_dict` and `model_state_dict` keys."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            checkpoint = {'state_dict': checkpoint}

        if logger:
            # Called once per shadow inside the 266-model logit-gen sweep;
            # noisy at INFO -- demote to DEBUG. Errors below stay at ERROR.
            logger.debug(f"Successfully loaded checkpoint from {checkpoint_path}")

        return model, checkpoint

    except Exception as e:
        if logger:
            logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
        raise


def validate_config(config, required_keys=None, logger=None):
    """Verify the config has every key in `required_keys` (dotted paths)."""
    if required_keys is None:
        required_keys = ['dataset.name']

    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                msg = f"Missing required config key: {key_path}"
                if logger:
                    logger.error(msg)
                raise ValueError(msg)
            current = current[key]

    if logger:
        logger.info("Configuration validation passed")

    return True


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, use_amp=True, logger=None, epoch=None, num_epochs=None, prefix="[Eval ]"):
    """Evaluate the model on a dataset; returns (avg_loss, accuracy_pct)."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        if use_amp:
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss_sum += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = loss_sum / len(data_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def aggregate_shadow_metrics(experiment_dir, num_models, logger=None):
    """
    Aggregate per-shadow-model metrics into one CSV.

    Reads `model_{i}/metrics.csv` for each model and writes
    `<experiment_dir>/shadow_metrics_aggregate.csv` with mean/std/count per metric.
    Loss_ratio and Acc_gap are recomputed from raw columns to avoid NaNs.
    """
    expdir = Path(experiment_dir)
    per_model = []

    def to_float(s):
        if s is None:
            return float('nan')
        s = str(s).strip().replace('%', '').replace('(', '').replace(')', '')
        try:
            return float(s)
        except ValueError:
            return float('nan')

    for i in range(num_models):
        p = expdir / f"model_{i}" / "metrics.csv"
        if not p.exists():
            if logger:
                logger.warning(f"[Aggregate] Missing {p}")
            continue
        with p.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
            if not rows:
                if logger:
                    logger.warning(f"[Aggregate] Empty {p}")
                continue
            r = rows[-1]
            tr_loss = to_float(r.get("Train_loss"))
            te_loss = to_float(r.get("Test_loss"))
            tr_acc = to_float(r.get("Train_acc (%)"))
            te_acc = to_float(r.get("Test_acc (%)"))

            loss_ratio = (te_loss / tr_loss) if (not math.isnan(tr_loss) and tr_loss != 0 and not math.isnan(te_loss)) else float('nan')
            acc_gap = (tr_acc - te_acc) if (not math.isnan(tr_acc) and not math.isnan(te_acc)) else float('nan')

            per_model.append({
                "Train_loss": tr_loss,
                "Test_loss": te_loss,
                "Loss_ratio": loss_ratio,
                "Train_acc (%)": tr_acc,
                "Test_acc (%)": te_acc,
                "Acc_gap (%)": acc_gap,
            })

    if not per_model:
        if logger:
            logger.error("[Aggregate] No metrics found.")
        return

    def mean(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return sum(xs) / len(xs) if xs else float('nan')

    def std(xs):
        xs = [x for x in xs if not math.isnan(x)]
        n = len(xs)
        if n < 2:
            return float('nan')
        m = sum(xs) / n
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))

    metrics = ["Train_loss", "Test_loss", "Loss_ratio", "Train_acc (%)", "Test_acc (%)", "Acc_gap (%)"]
    agg = {m: [pm[m] for pm in per_model] for m in metrics}

    out_path = expdir / "shadow_metrics_aggregate.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Mean", "Std", "Count"])
        for m in metrics:
            vals = agg[m]
            cnt = len([v for v in vals if not math.isnan(v)])
            w.writerow([m, f"{mean(vals):.4f}", f"{std(vals):.4f}", cnt])

    if logger:
        logger.info(f"[Aggregate] Saved aggregate metrics to {out_path}")
        for m in metrics:
            logger.info(f"{m:>15s} | mean={mean(agg[m]):.4f}  std={std(agg[m]):.4f}")
