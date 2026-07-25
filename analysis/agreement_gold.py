"""Agreement-based gold-set robustness (Pollock et al. Appendix C protocol).

Case B pool: N=10 WRN-28-2 targets trained on ONE fixed member/non-member
split (seed 20260718) with different training seeds, scored with LiRA-256
against the ORIGINAL experiment's 256-shadow pool (whose per-sample IN/OUT
masks are independent of the fixed split). Agreement gold sets are

    V^(p) = { member x : #{ i : x in V^(i) } >= ceil(p * N) },

for p in {0.5, 0.8, 1.0}, where V^(i) is target i's LiRA-256 vulnerable set
at FPR <= 1e-4. Methods (RARE, LT-IQR, LiRA-8) are then evaluated per target
against each V^(p) with the repo's capped P@k convention.

Stages (run in order; each is idempotent):
  --stage score    : LiRA-256 + LiRA-8 scores/thresholds for the new targets,
                     with validation against a stored original-target artifact.
  --stage analyze  : agreement sets, sanity checks, P@k table, CSV + LaTeX.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

import attacks as attacks_mod  # scripts/attacks.py
from rare.analysis.utils import (
    load_rare_locked_scores,
    score_avg_min_q_reference_losses,
    score_ltiqr_trajectory,
    tail_metrics_at_k,
)
from rare.attacks.signals import convert_signals

DEFAULT_POOL_EXP = ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140"
DEFAULT_NEW_EXP = ROOT / "experiments/cifar10/wrn28-2/agreement_2026-07-18"
ORIGINAL_TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
N_POOL_MODELS = 266
N_NEW = 10
FPR = 1e-4
LIRA_REF_SEED = 42          # attacks.py: lira_ref_seed defaults to config seed
K_FRACS = [0.01, 0.03, 0.05]
AGREEMENT_PS = [0.5, 0.8, 1.0]
DEFAULT_CSV = ROOT / "results/agreement_gold_cifar10.csv"
PAPER_RARE_P = {0.01: 0.95, 0.03: 0.83, 0.05: 0.72}  # CIFAR-10 headline row


def _labels_cifar10() -> np.ndarray:
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=str(ROOT / "data"), train=True, download=False)
    return np.asarray(ds.targets, dtype=np.int64)


def _lira_signal_from_logits(logits_path: Path, y: np.ndarray) -> np.ndarray:
    """(N, A) log-logit-scaling signal, matching attacks.py main()."""
    logits = np.load(logits_path).astype(np.float32)
    t = torch.from_numpy(logits)
    if t.ndim == 4:
        t = t[:, 0, :, :] if t.shape[1] == 1 else t.reshape(t.shape[0], -1, t.shape[-1])
    if t.ndim == 2:
        t = t[:, None, :]
    N, A, _ = t.shape
    labels_t = torch.from_numpy(y[:N]).long()
    out = np.empty((N, A), dtype=np.float32)
    for a in range(A):
        sig = convert_signals(t[:, a, :], labels_t, metric="log-logit-scaling", temp=1.0)
        out[:, a] = sig.cpu().numpy().reshape(-1)
    return out


def _threshold_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    """attacks.py compute_metrics convention: threshold of the last ROC point
    with fpr <= target."""
    fpr_list, _, thresholds = roc_curve(labels.astype(int), scores)
    idx = np.where(fpr_list <= fpr_target)[0]
    return float(thresholds[idx[-1]]) if idx.size else float("inf")


def stage_score(pool_exp: Path, new_exp: Path) -> None:
    y = _labels_cifar10()
    print("Loading 266 cached pool LiRA signals ...", flush=True)
    pool_signals = np.stack(
        [
            np.load(pool_exp / "signals" / f"lira_model_{i}.npy")
            for i in range(N_POOL_MODELS)
        ],
        axis=0,
    )  # (266, N, 2)
    keep_pool = np.load(pool_exp / "keep_indices.npy").astype(bool)
    keep_new = np.load(new_exp / "keep_indices.npy").astype(bool)
    n = pool_signals.shape[1]

    ref_pool = attacks_mod._build_reference_pool(N_POOL_MODELS, ORIGINAL_TARGETS)
    assert ref_pool.size == 256, ref_pool.size
    ref8 = attacks_mod._deterministic_choice(ref_pool, 8, seed=LIRA_REF_SEED)

    # ---- validation: reproduce stored original-target artifacts exactly ----
    # Pipeline convention (validated): 256 refs -> fix_variance=False;
    # low-budget 8 refs -> fix_variance=True (global-variance LiRA).
    configs = [(ref_pool, "lira_refs_256", False), (ref8, "lira_refs_8", True)]
    signals_t = torch.from_numpy(pool_signals)
    for ref_models, tag, fixvar in configs:
        got, _ = attacks_mod.lira_scores_for_model_fast(
            0, signals_t, keep_pool, fix_variance=fixvar, ref_models=ref_models
        )
        want = np.load(pool_exp / "model_0" / "attack_results" / f"{tag}_scores.npy")
        err = float(np.abs(got - want).max())
        print(f"  validation vs stored {tag} (original target 0): max|diff| = {err:.3e}")
        if err > 1e-3:
            raise SystemExit(f"VALIDATION FAILED for {tag}: max|diff|={err}")

    # ---- score the 10 new targets ----
    new_signals = np.stack(
        [
            _lira_signal_from_logits(new_exp / f"model_{j}" / "logits" / "logits.npy", y)
            for j in range(N_NEW)
        ],
        axis=0,
    )
    signals_all = torch.from_numpy(
        np.concatenate([pool_signals, new_signals], axis=0)
    )  # (276, N, 2)
    keep_all = np.concatenate([keep_pool, keep_new], axis=0)

    for j in range(N_NEW):
        tgt = N_POOL_MODELS + j
        out_dir = new_exp / f"model_{j}" / "attack_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        for ref_models, tag, fixvar in configs:
            scores, labels = attacks_mod.lira_scores_for_model_fast(
                tgt, signals_all, keep_all, fix_variance=fixvar, ref_models=ref_models
            )
            np.save(out_dir / f"{tag}_scores.npy", scores.astype(np.float32))
            np.save(out_dir / f"{tag}_labels.npy", labels.astype(np.int8))
            thr = {
                f: _threshold_at_fpr(labels, scores, f) for f in (0.0, 1e-4, 1e-3, 1e-2)
            }
            np.save(out_dir / f"{tag}_thresholds_by_fpr.npy", thr, allow_pickle=True)
            if tag == "lira_refs_256":
                vuln = (scores >= thr[FPR]) & labels.astype(bool)
                print(f"  new target {j}: |V| at FPR={FPR:g} -> {int(vuln.sum())}")
    print("stage score: done")


def _load_new_gold(new_exp: Path, j: int) -> tuple[np.ndarray, np.ndarray]:
    d = new_exp / f"model_{j}" / "attack_results"
    scores = np.load(d / "lira_refs_256_scores.npy").astype(np.float32)
    labels = np.load(d / "lira_refs_256_labels.npy").astype(bool)
    thr = np.load(d / "lira_refs_256_thresholds_by_fpr.npy", allow_pickle=True).item()
    return labels, (scores >= float(thr[FPR])) & labels


def _precisions(scores: np.ndarray, members: np.ndarray, vulnerable: np.ndarray) -> Dict[float, float]:
    pr = tail_metrics_at_k(scores, members, K_FRACS, vulnerable=vulnerable)
    return {kf: float(p) for kf, p in zip(K_FRACS, pr["precision"].to_numpy())}


def stage_analyze(pool_exp: Path, new_exp: Path, out_csv: Path) -> None:
    members = np.load(new_exp / "keep_indices.npy").astype(bool)[0]
    vuln_sets = []
    for j in range(N_NEW):
        labels, vuln = _load_new_gold(new_exp, j)
        if not np.array_equal(labels, members):
            raise RuntimeError(f"target {j} labels disagree with the fixed split")
        vuln_sets.append(vuln)
    counts = np.sum(vuln_sets, axis=0)  # n(x) over all 50k (0 outside members)

    union = counts >= 1
    inter = counts >= N_NEW
    agreement: Dict[float, np.ndarray] = {
        p: counts >= math.ceil(p * N_NEW) for p in AGREEMENT_PS
    }
    print("=== Gold-set sizes ===")
    print(f"  per-target |V^(i)|: {[int(v.sum()) for v in vuln_sets]}")
    print(f"  union: {int(union.sum())}  intersection: {int(inter.sum())}")
    for p, v in agreement.items():
        print(f"  |V^({p})| (n >= {math.ceil(p*N_NEW)}): {int(v.sum())}")
    assert inter.sum() <= agreement[0.5].sum() <= union.sum()

    # ---- per-target method scores ----
    method_scores: Dict[str, List[np.ndarray]] = {
        "RARE": [], "LT-BTM-R": [], "LT-IQR": [], "LiRA-8": []
    }
    for j in range(N_NEW):
        method_scores["RARE"].append(load_rare_locked_scores(new_exp, j, num_refs=1, q=0.5))
        traj = np.load(new_exp / f"model_{j}" / "per_epoch_losses.npy").astype(np.float32).T
        method_scores["LT-IQR"].append(score_ltiqr_trajectory(traj))
        ref_p = sorted((new_exp / f"model_{j}").glob("ref_losses_E*_*_ref0.npy"))[0]
        method_scores["LT-BTM-R"].append(
            score_avg_min_q_reference_losses([np.load(ref_p).astype(np.float32)], q=0.5)
        )
        method_scores["LiRA-8"].append(
            np.load(new_exp / f"model_{j}" / "attack_results" / "lira_refs_8_scores.npy")
        )

    # ---- sanity: RARE vs own-target V^(i) should match paper numbers ----
    own = np.array([
        list(_precisions(method_scores["RARE"][j], members, vuln_sets[j]).values())
        for j in range(N_NEW)
    ])  # (10, 3)
    own_mean = own.mean(axis=0)
    print("\n=== Sanity: RARE vs own-target V^(i) (paper: 0.95 / 0.83 / 0.72) ===")
    print(f"  P@1/3/5 = {own_mean[0]:.3f} / {own_mean[1]:.3f} / {own_mean[2]:.3f}")
    for kf, got in zip(K_FRACS, own_mean):
        if abs(got - PAPER_RARE_P[kf]) > 0.06:
            raise SystemExit(
                f"SANITY FAILURE: RARE own-gold P@{int(kf*100)}%={got:.3f} vs "
                f"paper {PAPER_RARE_P[kf]:.2f} — indexing bug likely."
            )

    # ---- main table: methods x p x k ----
    rows = []
    for method, per_target in method_scores.items():
        for p, vp in agreement.items():
            per = np.array([
                list(_precisions(per_target[j], members, vp).values())
                for j in range(N_NEW)
            ])
            rows.append({
                "method": method, "p": p, "gold_size": int(vp.sum()),
                **{f"p{int(k*100)}_mean": float(per[:, i].mean()) for i, k in enumerate(K_FRACS)},
                **{f"p{int(k*100)}_std": float(per[:, i].std(ddof=1)) for i, k in enumerate(K_FRACS)},
            })
        # single-model reference row (mean over targets vs their own V^(i))
        per_own = np.array([
            list(_precisions(per_target[j], members, vuln_sets[j]).values())
            for j in range(N_NEW)
        ])
        rows.append({
            "method": method, "p": "own (N=1)",
            "gold_size": int(np.mean([v.sum() for v in vuln_sets])),
            **{f"p{int(k*100)}_mean": float(per_own[:, i].mean()) for i, k in enumerate(K_FRACS)},
            **{f"p{int(k*100)}_std": float(per_own[:, i].std(ddof=1)) for i, k in enumerate(K_FRACS)},
        })
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Pollock-shape flag: does precision rise from own-gold to agreement gold?
    print("\n=== Direction vs Pollock et al. (LT-IQR rose 0.92 -> 0.96 at p=0.5, P@1%) ===")
    for method in method_scores:
        own_p1 = float(df[(df.method == method) & (df.p == "own (N=1)")]["p1_mean"].iloc[0])
        agr_p1 = float(df[(df.method == method) & (df.p == 0.5)]["p1_mean"].iloc[0])
        arrow = "ROSE" if agr_p1 > own_p1 else "FELL"
        print(f"  {method}: P@1% {own_p1:.3f} -> {agr_p1:.3f} against V^(0.5): {arrow}")

    # ---- LaTeX table ----
    def fmt(mean: float, std: float) -> str:
        return f"{mean:.2f}{{\\scriptstyle\\pm{std:.2f}}}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Agreement-based gold-set robustness on CIFAR-10 / WRN-28-2:"
        r" capped P@$k$\% (mean$\pm$std over 10 fixed-split targets) against"
        r" agreement gold sets $V^{(p)}$ from 10 LiRA-256 audits at FPR$\leq 10^{-4}$.}",
        r"\label{tab:agreement-gold}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l l r ccc}",
        r"\toprule",
        r"\textbf{Method} & $p$ & $|V^{(p)}|$ & \textbf{P@1\%} & \textbf{P@3\%} & \textbf{P@5\%} \\",
        r"\midrule",
    ]
    for method in ["LiRA-8", "LT-IQR", "LT-BTM-R", "RARE"]:
        for p in ["own (N=1)"] + AGREEMENT_PS:
            r_ = df[(df.method == method) & (df.p == p)].iloc[0]
            p_label = "single" if p == "own (N=1)" else f"{p:g}"
            lines.append(
                f"{method} & {p_label} & {int(r_['gold_size'])} & "
                + " & ".join(fmt(r_[f"p{k}_mean"], r_[f"p{k}_std"]) for k in (1, 3, 5))
                + r" \\"
            )
        if method != "RARE":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = out_csv.with_suffix(".tex")
    tex_path.write_text("\n".join(lines) + "\n")

    meta = {
        "protocol": "Pollock et al. Appendix C Def. C.1 (agreement gold)",
        "regime": "Case B: 10 targets, one fixed split (seed 20260718), seeds 43..52",
        "gold": "LiRA-256 vs original 256-shadow pool, FPR<=1e-4",
        "sizes": {
            "per_target": [int(v.sum()) for v in vuln_sets],
            "union": int(union.sum()), "intersection": int(inter.sum()),
            **{f"V^{p}": int(v.sum()) for p, v in agreement.items()},
        },
    }
    out_csv.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out_csv}\nwrote {tex_path}\nwrote {out_csv.with_suffix('.meta.json')}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["score", "analyze"], required=True)
    parser.add_argument("--pool-exp", type=Path, default=DEFAULT_POOL_EXP)
    parser.add_argument("--new-exp", type=Path, default=DEFAULT_NEW_EXP)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    if args.stage == "score":
        stage_score(args.pool_exp.resolve(), args.new_exp.resolve())
    else:
        stage_analyze(args.pool_exp.resolve(), args.new_exp.resolve(), args.output_csv.resolve())


if __name__ == "__main__":
    main()
