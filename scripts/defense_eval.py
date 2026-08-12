"""Evaluate the RARE-guided logit-replacement defense against SotA MIAs.

Defense (locked): for each target, rank members by RARE (M=1, q=50%),
percentile-normalize over members, and binary-replace the top-k% most
vulnerable members' logits -- across the full A=2 augmentation grid --
with the logits of the single OUT reference model (ref0, trained on data
disjoint from the target's members). Everything else is untouched.

For each benchmark and each strong attack (LiRA-256, RMIA-256,
Attack-R-128) we recompute the target's attack signal from the defended
logits, re-run the attack with the reference pool unchanged, and report
AUC, TPR@1e-3, TPR@1e-4 before vs after, plus the same on the protected
member subset. The defense touches only the target's served member
logits; shadow/reference distributions are fixed, so only the target row
of the signal tensor changes.

Read-only w.r.t. the experiment dirs (nothing is written back).

Usage:
  python scripts/defense_eval.py
  python scripts/defense_eval.py --k 0.10 --benchmarks cifar10
"""

import _bootstrap  # noqa: F401

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import rankdata
from sklearn.metrics import auc as sk_auc, roc_curve
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import attacks as A  # scripts/attacks.py
from rare.analysis.utils import load_rare_locked_scores
from rare.attacks.signals import convert_signals
from rare.utils.common import load_checkpoint_safe
from rare.utils.data import load_dataset, resolve_subsample
from rare.utils.logits import LogitsGenerator
from rare.utils.model import get_model

BENCHMARKS = {
    "cifar10": ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140",
    "cifar100": ROOT / "experiments/cifar100/wrn28-2/2025-12-25_0121",
    "cinic10": ROOT / "experiments/cinic10/wrn28-2/2026-05-11_0215",
}
TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
FPRS = [1e-3, 1e-4, 1e-5]
REF0_RE = re.compile(r"^ref_losses_E(\d+)_(.+)_ref0\.npy$")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
log = logging.getLogger("defense")
logging.basicConfig(level=logging.WARNING)


def make_generator(cfg, nb_augmentation):
    """A LogitsGenerator at the given augmentation, so ref0 logits are produced
    by the exact pipeline forward pass (inference_mode, batch_size, folded-batch
    BN) that generated the cached signals. A=2 for LiRA, A=1 (nb=0) otherwise."""
    c = yaml.safe_load(yaml.safe_dump(cfg))   # deep copy
    c["inference_data_augmentations"] = {"mode": "lira_aug",
                                         "nb_augmentation": int(nb_augmentation)}
    c.setdefault("logits_generation", {})["overwrite"] = True
    return LogitsGenerator(c, log)


def gen_logits(gen, ckpt):
    """Load `ckpt` into the generator and run its forward pass -> (N, A, C)."""
    gen._inner_model, _ = load_checkpoint_safe(str(ckpt), gen._inner_model,
                                               gen.device, log)
    gen.model = gen._inner_model
    return gen._get_model_logits().cpu()


def lira_sig(logits_NAC, labels_t):
    out = torch.empty(logits_NAC.shape[0], logits_NAC.shape[1], dtype=torch.float32)
    for a in range(logits_NAC.shape[1]):
        out[:, a] = convert_signals(logits_NAC[:, a, :], labels_t, "log-logit-scaling", 1.0)
    return out


def rmia_signal_spec(exp):
    """(metric, temp, extra, n_views) for RMIA, as attacks.py builds it.

    attacks.py maps the config's `rmia.signal` name onto a convert_signals
    metric plus extras, applies `rmia.temperature`, and writes one column per
    augmentation view (`rmia.nb_augmentation`). Reproduce all three here or
    the cached signals cannot be matched.
    """
    r = yaml.safe_load(open(exp / "attack_config.yaml"))["rmia"]
    name = r.get("signal", "target_logit")
    if name == "softmax_relative":
        metric, extra = "softmax", None
    elif name == "taylor_softmax_relative":
        metric, extra = "taylor", {"taylor_n": r["taylor_n"]}
    elif name == "sm_softmax_relative":
        metric, extra = "soft-margin", {"taylor_m": r["taylor_m"]}
    elif name == "sm_taylor_softmax_relative":
        metric, extra = ("taylor-soft-margin",
                         {"taylor_m": r["taylor_m"], "taylor_n": r["taylor_n"]})
    else:
        metric, extra = name, None
    return metric, float(r.get("temperature", 1.0)), extra, int(r.get("nb_augmentation", 1))


def rmia_sig(logits_NAC, labels_t, metric, temp, extra, n_views):
    """RMIA signal, one column per augmentation view (matches attacks.py)."""
    a_use = min(int(n_views), logits_NAC.shape[1])
    out = torch.empty(logits_NAC.shape[0], a_use, dtype=torch.float32)
    for a in range(a_use):
        out[:, a] = convert_signals(
            logits_NAC[:, a, :], labels_t, metric=metric, temp=temp, extra=extra)
    return out


def attackr_sig(logits_NAC, y_np):
    # Attack-R signal = CE loss from logits, view 0.
    return torch.from_numpy(
        A.ce_loss_from_logits(logits_NAC[:, 0, :].numpy(), y_np).astype(np.float32)
    ).reshape(-1, 1)


def metrics(scores, labels, prot_idx):
    fpr, tpr, thr = roc_curve(labels.astype(int), scores)
    roc_auc, _, tprd, thrd = A.compute_metrics(fpr, tpr, thr, FPRS)
    prot = {f: float((scores[prot_idx] >= thrd[f]).mean()) for f in FPRS}
    return roc_auc, tprd, prot


def run_attack(name, signals, keep, ref_pool, t):
    """Return (scores, labels) for the chosen strong-attack config."""
    if name == "LiRA-256":
        return A.lira_scores_for_model_fast(t, signals, keep, False, ref_models=ref_pool)
    if name == "RMIA-256":
        s, l = A.rmia_scores_for_model_gpu(
            t, signals, keep, ref_models=ref_pool, gamma=2.0, proportiontocut=0.0,
            top_k=-1, num_ref_models=2 * 256, ref_selection_seed=42,
            offline=False, offline_a=0.3, augmentation_mode="original",
            device=DEV, chunk_size=5000)
        return s.astype(np.float32), l
    if name == "Attack-R-128":
        sub = A._deterministic_choice(ref_pool, 128, seed=42)
        s, l = A.attackr_scores_for_model_fast(
            t, signals, keep, ref_models=sub, num_ref_models=-1, ref_selection_seed=42)
        return s.astype(np.float32), l
    raise ValueError(name)


SIG_FN = {"LiRA-256": "lira", "RMIA-256": "rmia", "Attack-R-128": "attackr"}


def evaluate_benchmark(name, exp, taus, targets):
    cfg = yaml.safe_load(open(exp / "logit_config.yaml"))
    keep = np.load(exp / "keep_indices.npy").astype(bool)
    M, N = keep.shape
    ref_pool = A._build_reference_pool(M, targets)

    # One pipeline generator at A=2. RMIA/Attack-R use view 0 of it (which is
    # exactly what attacks.py slices for them) -- NOT a separate A=1 pass.
    gen2 = make_generator(cfg, 2)
    labels_t = torch.cat(
        [y for _, y in DataLoader(gen2.full_dataset, batch_size=4096, shuffle=False)]
    ).long()
    y_np = labels_t.numpy()
    assert labels_t.shape[0] == N, (labels_t.shape, N)

    # Cached signals (M, N, A) for the three attacks.
    sig = {}
    for short in ("lira", "rmia", "attackr"):
        files = [np.load(exp / "signals" / f"{short}_model_{i}.npy") for i in range(M)]
        sig[short] = torch.from_numpy(np.stack(files).astype(np.float32))

    # agg[tau][attack] = {base, def, prot}; util[tau] = list of (acc_b, acc_a).
    agg = {tau: {at: {"base": [], "def": [], "prot": []} for at in SIG_FN}
           for tau in taus}
    util = {tau: [] for tau in taus}
    sanity = []

    rmia_metric, rmia_temp, rmia_extra, rmia_views = rmia_signal_spec(exp)
    sig_fn = {"lira": lambda L: lira_sig(L, labels_t),
              "rmia": lambda L: rmia_sig(L, labels_t, rmia_metric, rmia_temp,
                                         rmia_extra, rmia_views),
              "attackr": lambda L: attackr_sig(L, y_np)}

    for t in targets:
        td = exp / f"model_{t}"
        mem = keep[t]; mem_idx = np.where(mem)[0]
        rare = load_rare_locked_scores(exp, t, num_refs=1, q=0.50)
        r = rare[mem].astype(np.float32)

        tag = next(REF0_RE.match(p.name).group(2)
                   for p in sorted(td.glob("ref_losses_E*_*_ref0.npy"))
                   if REF0_RE.match(p.name))
        ref_ckpt = td / f"ref_model_{tag}_ref0.pth"

        # Target A=2 logits from the cached file; ref0 A=2 logits from the
        # cached ref_logits_*.npy (written by scripts/save_reference_logits.py
        # during the RARE step), falling back to an on-the-fly forward.
        tl2 = torch.from_numpy(np.asarray(
            np.load(td / "logits" / "logits.npy"), dtype=np.float32))
        if tl2.ndim == 4:
            tl2 = tl2[:, 0]                                  # (N, 2, C)
        ref_logits_p = next(td.glob(f"ref_logits_E*_{tag}_ref0.npy"), None)
        if ref_logits_p is not None:
            rl2 = torch.from_numpy(np.load(ref_logits_p).astype(np.float32))
        else:
            rl2 = gen_logits(gen2, ref_ckpt)                 # (N, 2, C)

        # LiRA and RMIA use both A=2 views; Attack-R uses view 0 only (as attacks.py).
        base_basis = {"lira": tl2, "rmia": tl2, "attackr": tl2[:, :1, :]}
        ref_basis = {"lira": rl2, "rmia": rl2, "attackr": rl2[:, :1, :]}

        # Baseline (tau-independent) per attack: cached signal + sanity gate.
        base = {}
        row_sanity = {}
        for at, short in SIG_FN.items():
            row_sanity[at] = float(
                (sig_fn[short](base_basis[short]) - sig[short][t]).abs().max())
            b_scores, b_labels = run_attack(at, sig[short], keep, ref_pool, t)
            base[at] = (b_labels, metrics(b_scores, b_labels, mem_idx))
        sanity.append(row_sanity)
        tl_arg = tl2[:, 0].argmax(1).numpy()

        for tau in taus:
            # alpha = sigmoid(RARE / tau) over all members.
            alpha = torch.from_numpy(1.0 / (1.0 + np.exp(-r / tau)))
            a = alpha[:, None, None]
            # Utility: member serving accuracy (view 0) before/after; blend the
            # view-0 logits and take the argmax.
            dv0 = (1 - a[:, 0]) * tl2[mem_idx, 0] + a[:, 0] * rl2[mem_idx, 0]
            acc_b = float((tl_arg[mem_idx] == y_np[mem_idx]).mean())
            acc_a = float((dv0.argmax(1).numpy() == y_np[mem_idx]).mean())
            util[tau].append((acc_b, acc_a))

            for at, short in SIG_FN.items():
                b_labels, (b_auc, b_tpr, b_prot) = base[at]
                orig_row = sig[short][t]
                tl_b, rl_b = base_basis[short], ref_basis[short]
                dl_b = tl_b.clone()
                dl_b[mem_idx] = (1 - a) * tl_b[mem_idx] + a * rl_b[mem_idx]
                def_sig = sig_fn[short](dl_b)
                def_row = orig_row.clone()
                def_row[mem_idx] = def_sig[mem_idx]
                sig[short][t] = def_row
                d_scores, _ = run_attack(at, sig[short], keep, ref_pool, t)
                d_auc, d_tpr, d_prot = metrics(d_scores, b_labels, mem_idx)
                sig[short][t] = orig_row

                agg[tau][at]["base"].append((b_auc, b_tpr[1e-3], b_tpr[1e-4], b_tpr[1e-5]))
                agg[tau][at]["def"].append((d_auc, d_tpr[1e-3], d_tpr[1e-4], d_tpr[1e-5]))
                agg[tau][at]["prot"].append((b_prot[1e-3], b_prot[1e-4],
                                             d_prot[1e-3], d_prot[1e-4]))
        print(f"  [{name}] target {t}: sanity max|Δsig| "
              f"{max(row_sanity.values()):.2e}", flush=True)

    return agg, util, sanity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau", type=float, nargs="*", default=[2.0],
                    help="sigmoid temperature(s); 2.0 is the locked setting")
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    ap.add_argument("--targets", type=int, nargs="*", default=TARGETS)
    args = ap.parse_args()
    taus = list(args.tau)

    print("Defense: soft per-member logit blend (1-a)*target + a*ref0, "
          f"a = sigmoid(RARE/tau) over ALL members. tau sweep={taus}. "
          f"Device={DEV}\n")
    for name in args.benchmarks:
        t0 = time.time()
        agg, util, sanity = evaluate_benchmark(name, BENCHMARKS[name], taus, args.targets)
        max_sanity = max(max(s.values()) for s in sanity)
        if max_sanity > 1e-3:
            # Same hard gate as tab_defense.py: if the undefended pipeline does
            # not reproduce the cached signals, every before/after number below
            # is meaningless, so refuse to print them.
            raise SystemExit(
                f"SANITY GATE FAILED for {name}: max|Δsignal| = {max_sanity:.2e} "
                f"(tolerance 1e-3). Recomputed attack signals do not match "
                f"<exp>/signals/*.npy; results suppressed."
            )
        print(f"\n===== {name.upper()} (all members, {time.time()-t0:.0f}s, "
              f"baseline sanity max|Δsig|={max_sanity:.1e}) =====")
        for tau in taus:
            ub = np.mean([u[0] for u in util[tau]])
            ua = np.mean([u[1] for u in util[tau]])
            print(f"\n-- tau={tau:g} | member serving acc: {ub:.4f} -> {ua:.4f}")
            hdr = (f"{'attack':<14}{'AUC b/a':>14}{'TPR1e-3 b/a':>20}"
                   f"{'TPR1e-4 b/a':>20}{'TPR1e-5 b/a':>20}")
            print(hdr)
            for at in SIG_FN:
                b = np.array(agg[tau][at]["base"]); d = np.array(agg[tau][at]["def"])
                print(f"{at:<14}"
                      f"{b[:,0].mean():>6.2f}/{d[:,0].mean():<7.2f}"
                      f"{b[:,1].mean():>9.4f}/{d[:,1].mean():<10.4f}"
                      f"{b[:,2].mean():>9.4f}/{d[:,2].mean():<10.4f}"
                      f"{b[:,3].mean():>9.4f}/{d[:,3].mean():<10.4f}")


if __name__ == "__main__":
    main()
