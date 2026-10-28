"""Measure the per-target timings behind the runtime table (tab:pipeline-cost).

`paper_tabs_figs/tab_pipeline_cost.py` combines training-time phases that the
pipeline logs itself (`runtime.csv`: train, target_train_*, ref_retrain,
lt_iqr_score, liu_mlp_score) with the stand-alone inference timings measured
here:

  --stage logits      per-model logit generation over the training pool with
                      A=1 and A=2 views              -> logit_gen_runtime.csv
                      one A=1 forward pass + CE loss (LOSS attack)
                                                     -> loss_forward_pass_runtime.csv
  --stage attacks     cold per-target attack scoring from cached logits:
                      load the target's and the M references' logits, convert
                      them to the attack signal, score
                                                     -> score_inference_runtime.csv
  --stage trajectory  RARE score computation from the on-disk trajectories,
                      appended to runtime.csv as
                        anomaly_ltiqr  LT-IQR from the target trajectory
                        rare_ltbtmr    LT-BTM-R from the reference trajectory
                        anomaly_rare   the standardized sum z_T(.) + z_T(.)
  --stage all         all three

Each timing is wall-clock with CUDA synchronized, after one untimed warm-up,
on the same reference subsets and settings as scripts/attacks.py.

Output goes to <exp>/runtime_measurements/ by default, so the timings shipped
with an experiment are never overwritten; regenerate the table from a new set
with `python paper_tabs_figs/tab_pipeline_cost.py --timings-dir <dir>`.

Usage:
  python scripts/measure_runtime.py --config configs/cifar10_wrn28-2/attack_config.yaml --stage all
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import attacks as A  # scripts/attacks.py
from rare.analysis.utils import (
    _zscore_ignore_nan, score_avg_min_q_reference_losses, score_ltiqr_trajectory)
from rare.attacks.signals import convert_signals
from rare.utils.common import append_runtime, load_checkpoint_safe
from rare.utils.data import load_dataset, resolve_subsample

DEFAULT_ATTACKS = "LiRA:256,LiRA:8,RMIA:8,Attack-R:16,LOSS:0"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def _labels(cfg: dict, exp: Path) -> np.ndarray:
    """Training-pool labels in keep_indices order (subsample applied)."""
    ds, _, _ = load_dataset(cfg, mode="inference")
    y = np.asarray(ds.targets)
    d = cfg.get("dataset", {})
    sub = resolve_subsample(len(y), d.get("n_data_points"), int(d.get("subsample_seed", cfg.get("seed", 42))),
                            save_path=str(exp / "subsample_indices.npy"))
    return y[sub] if sub.size != len(y) else y


def _load_logits(exp: Path, i: int) -> torch.Tensor:
    """Cached logits of model i as an (N, A, C) float32 tensor."""
    x = torch.from_numpy(np.asarray(np.load(exp / f"model_{i}" / "logits" / "logits.npy"), dtype=np.float32))
    if x.ndim == 4:
        x = x[:, 0] if x.shape[1] == 1 else x.reshape(x.shape[0], -1, x.shape[-1])
    return x[:, None, :] if x.ndim == 2 else x


# ---------------------------------------------------------------------------
# Stage: logits
# ---------------------------------------------------------------------------
def stage_logits(logit_cfg: dict, exp: Path, targets, out_dir: Path):
    from rare.utils.logits import LogitsGenerator

    logger = logging.getLogger("measure_runtime")
    gen = LogitsGenerator(logit_cfg, logger)
    tm = gen.target_model
    ckpt_name = "best_model.pth" if str(tm) == "best" else f"checkpoint_epoch{tm}.pth"
    y = torch.as_tensor(_labels(logit_cfg, exp), device=gen.device).long()

    def timed_logits(t: int, n_views: int) -> tuple:
        gen.spatial_shift, gen.use_horizontal_flip = 0, n_views == 2
        _sync(); t0 = time.perf_counter()
        load_checkpoint_safe(str(exp / f"model_{t}" / ckpt_name), gen._inner_model, gen.device, logger)
        z = gen._get_model_logits()
        _sync()
        return z, time.perf_counter() - t0

    timed_logits(targets[0], 2)                                   # warm-up
    gen_rows, loss_rows = [], []
    for t in targets:
        for n_views in (1, 2):
            z, sec = timed_logits(t, n_views)
            z.cpu().numpy()
            gen_rows.append([t, sec, n_views])
        # LOSS: one A=1 forward pass plus the per-sample cross-entropy.
        z, sec = timed_logits(t, 1)
        _sync(); t0 = time.perf_counter()
        torch.nn.functional.cross_entropy(z[:, 0], y[: z.shape[0]], reduction="none").cpu().numpy()
        _sync()
        loss_rows.append([t, sec + time.perf_counter() - t0])
        print(f"  logits target {t}: A=1 {gen_rows[-2][1]:.2f}s  A=2 {gen_rows[-1][1]:.2f}s")
    _write_csv(out_dir / "logit_gen_runtime.csv", ["target", "seconds", "num_augs"], gen_rows)
    _write_csv(out_dir / "loss_forward_pass_runtime.csv", ["target", "seconds"], loss_rows)


# ---------------------------------------------------------------------------
# Stage: attacks
# ---------------------------------------------------------------------------
def _signal(method: str, logits: torch.Tensor, labels: torch.Tensor, cfg: dict) -> torch.Tensor:
    """(N, A) attack signal of one model, as computed by scripts/attacks.py."""
    if method == "LiRA":
        n_aug = int(cfg["lira"].get("nb_augmentation", 18))
        return torch.stack([convert_signals(logits[:, a], labels, metric="log-logit-scaling", temp=1.0)
                            .reshape(-1).cpu() for a in range(min(n_aug, logits.shape[1]))], dim=1)
    if method == "RMIA":
        r = cfg["rmia"]
        name = str(r.get("signal", "sm_taylor_softmax_relative"))
        extra = {"taylor_m": float(r.get("taylor_m", 0.6)), "taylor_n": int(r.get("taylor_n", 4))}
        metric, ex = {"softmax_relative": ("softmax", None),
                      "taylor_softmax_relative": ("taylor", {"taylor_n": extra["taylor_n"]}),
                      "sm_softmax_relative": ("soft-margin", {"taylor_m": extra["taylor_m"]}),
                      "sm_taylor_softmax_relative": ("taylor-soft-margin", extra)}.get(name, (name, extra))
        n_aug = int(r.get("nb_augmentation", 18))
        return torch.stack([convert_signals(logits[:, a], labels, metric=metric,
                                            temp=float(r.get("temperature", 2.0)), extra=ex)
                            .reshape(-1).cpu() for a in range(min(n_aug, logits.shape[1]))], dim=1)
    if method == "Attack-R":
        n_aug = int(cfg["attackr"].get("nb_augmentation", 1))
        y = labels.numpy()
        return torch.stack([torch.from_numpy(A.ce_loss_from_logits(logits[:, a].numpy(), y).astype(np.float32))
                            for a in range(min(n_aug, logits.shape[1]))], dim=1)
    raise ValueError(method)


def _score(method: str, M: int, sig: torch.Tensor, keep: np.ndarray, cfg: dict):
    """Score target 0 of the compact (1+M)-model arrays with references 1..M."""
    refs = np.arange(1, M + 1)
    if method == "LiRA":
        fix = bool(cfg["lira"].get("fix_variance", False)) or M < 64
        return A.lira_scores_for_model_fast(0, sig, keep, fix, ref_models=refs)
    if method == "Attack-R":
        return A.attackr_scores_for_model_fast(0, sig, keep, ref_models=refs, num_ref_models=-1,
                                               ref_selection_seed=int(cfg["attackr"].get("ref_selection_seed", 42)))
    r = cfg["rmia"]
    fn = A.rmia_scores_for_model_gpu if r.get("use_gpu", False) else A.rmia_scores_for_model
    kw = {"device": "cuda", "chunk_size": int(r.get("gpu_chunk_size", 5000))} if r.get("use_gpu", False) else {}
    return fn(0, sig, keep, ref_models=refs, gamma=float(r.get("gamma", 2.0)),
              proportiontocut=float(r.get("proportiontocut", 0.2)), top_k=int(r.get("top_k", -1)),
              num_ref_models=2 * M, ref_selection_seed=int(r.get("ref_selection_seed", 42)),
              offline=r.get("offline", False), offline_a=float(r.get("offline_a", 0.3)),
              augmentation_mode=str(r.get("augmentation", "augmented")).lower(), **kw)


def stage_attacks(cfg: dict, exp: Path, targets, out_dir: Path, attack_list):
    labels_np = _labels(cfg, exp)
    labels = torch.from_numpy(labels_np).long()
    n_models = int(cfg.get("num_shadow_models", cfg.get("training", {}).get("num_shadow_models", 256)))
    keep_all = np.load(exp / "keep_indices.npy")[:n_models, :labels_np.size].astype(bool)
    pool = A._build_reference_pool(n_models, [int(x) for x in cfg["target_models"]])
    section = {"LiRA": "lira", "RMIA": "rmia", "Attack-R": "attackr"}
    dataset = str(cfg["dataset"]["name"]).lower()

    def run(method, M, t):
        if method == "LOSS":
            t0 = time.perf_counter()
            z = _load_logits(exp, t)[: labels_np.size].numpy()
            t1 = time.perf_counter()
            A.loss_scores_for_model(0, z, labels_np, keep_all[[t]],
                                    nb_augmentation=int(cfg.get("loss", {}).get("nb_augmentation", 1)))
            t2 = time.perf_counter()
            return t1 - t0, 0.0, t2 - t1
        refs = pool if M >= pool.size else A._deterministic_choice(
            pool, M, seed=int(cfg[section[method]].get("ref_selection_seed", cfg.get("seed", 42))))
        models = [int(t)] + [int(r) for r in refs]
        t_load = t_sig = 0.0
        sigs = []
        for i in models:              # convert model by model: logits never all in memory
            t0 = time.perf_counter()
            z = _load_logits(exp, i)[: labels_np.size]
            t1 = time.perf_counter()
            sigs.append(_signal(method, z, labels[: z.shape[0]], cfg))
            t_load += t1 - t0
            t_sig += time.perf_counter() - t1
        sig = torch.stack(sigs, dim=0)
        _sync(); t0 = time.perf_counter()
        _score(method, M, sig, keep_all[models], cfg)
        _sync()
        return t_load, t_sig, time.perf_counter() - t0

    run(attack_list[0][0], attack_list[0][1], targets[0])          # warm-up
    rows = []
    for method, M in attack_list:
        for t in targets:
            tl, ts, tc = run(method, M, t)
            rows.append([dataset, method, M, t, tl, ts, tc, tl + ts + tc])
            print(f"  {method:<8} M={M:<3} target {t}: {tl + ts + tc:.2f}s")
    _write_csv(out_dir / "score_inference_runtime.csv",
               ["dataset", "method", "M", "target", "t_load", "t_signal", "t_score", "t_total"], rows)


# ---------------------------------------------------------------------------
# Stage: trajectory
# ---------------------------------------------------------------------------
def stage_trajectory(cfg: dict, exp: Path, targets, out_dir: Path, q: float = 0.5):
    keep = np.load(exp / "keep_indices.npy").astype(bool)
    csv_path = out_dir / "runtime.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = {"arch": cfg["model"]["architecture"], "dataset": cfg["dataset"]["name"]}

    def run(t):
        td = exp / f"model_{t}"
        members = keep[t]
        t0 = time.perf_counter()
        s_t = score_ltiqr_trajectory(np.load(td / "per_epoch_losses.npy").astype(np.float32).T)
        t1 = time.perf_counter()
        ref = np.load(sorted(td.glob("ref_losses_E*_*_ref0.npy"))[0]).astype(np.float32)
        s_r = score_avg_min_q_reference_losses([ref], q=q)
        t2 = time.perf_counter()
        _zscore_ignore_nan(s_t, members) + _zscore_ignore_nan(s_r, members)
        t3 = time.perf_counter()
        return t1 - t0, t2 - t1, t3 - t2

    run(targets[0])                                               # warm-up
    for t in targets:
        a, b, c = run(t)
        for phase, sec in (("anomaly_ltiqr", a), ("rare_ltbtmr", b), ("anomaly_rare", c)):
            append_runtime(str(csv_path), phase=phase, item=f"target_{int(t)}", seconds=sec, **extra)
        print(f"  RARE score target {t}: LT-IQR {a:.3f}s  LT-BTM-R {b:.3f}s  sum {c:.4f}s")
    print(f"appended to {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path,
                    help="Benchmark attack_config.yaml (logit_config.yaml is read from the same folder).")
    ap.add_argument("--stage", choices=["logits", "attacks", "trajectory", "all"], default="all")
    ap.add_argument("--targets", type=int, nargs="*", default=None)
    ap.add_argument("--attacks", default=DEFAULT_ATTACKS,
                    help=f"Comma-separated method:M pairs (default {DEFAULT_ATTACKS}).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: <experiment_dir>/runtime_measurements")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    exp = Path(cfg["experiment"]["checkpoint_dir"])
    train_cfg_p = exp / "train_config.yaml"
    if train_cfg_p.exists():                                       # as scripts/attacks.py does
        A.recursive_update(cfg, {k: v for k, v in yaml.safe_load(open(train_cfg_p)).items()
                                 if k not in cfg})
    targets = args.targets or [int(t) for t in cfg["target_models"]]
    out_dir = args.out_dir or exp / "runtime_measurements"
    torch.manual_seed(0)
    np.random.seed(0)

    if args.stage in ("logits", "all"):
        logit_cfg = yaml.safe_load(open(args.config.parent / "logit_config.yaml"))
        stage_logits(logit_cfg, exp, targets, out_dir)
    if args.stage in ("attacks", "all"):
        attack_list = [(m, int(k)) for m, k in (p.split(":") for p in args.attacks.split(","))]
        stage_attacks(cfg, exp, targets, out_dir, attack_list)
    if args.stage in ("trajectory", "all"):
        stage_trajectory(cfg, exp, targets, out_dir)


if __name__ == "__main__":
    main()
