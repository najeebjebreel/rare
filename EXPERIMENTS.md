# Reproduction Guide

How to reproduce the results in the paper from this repository. Every table
and figure is regenerated from on-disk artifacts by a single script in
`paper_tabs_figs/`; the sections below describe how to produce those artifacts.

## 1. Environment and data

```bash
conda create -n rare python=3.10 && conda activate rare
pip install -e .
```

CIFAR-10/100 download automatically to `data/`. For CINIC-10, place the
official `train/`, `valid/`, `test/` folders under `data/cinic10/` and run:

```bash
python scripts/build_lmdb_cinic.py
```

Benchmarks (WRN-28-2 targets) live in dedicated experiment directories:

| Benchmark | Experiment directory |
|---|---|
| CIFAR-10  | `experiments/cifar10/wrn28-2/2025-12-07_0140` |
| CIFAR-100 | `experiments/cifar100/wrn28-2/2025-12-25_0121` |
| CINIC-10  | `experiments/cinic10/wrn28-2/2026-05-11_0215` |

Cross-architecture results (Table 3) additionally use WRN-40-4 and ResNet-20
targets on CIFAR-10. All benchmarks use the ten targets
`[0, 25, 50, 75, 100, 125, 150, 175, 200, 250]`.

## 2. Pipeline

### 2a. Gold standard (one-time prerequisite, compute-heavy)

Each benchmark trains a pool of 266 models, caches their logits, and runs the
reference attacks that define the gold vulnerable sets and the reference-based
baseline rows (LiRA / RMIA / Attack-R / LOSS):

```bash
EXP=experiments/cifar10/wrn28-2/2025-12-07_0140
python scripts/train.py          --config "$EXP/train_config.yaml"          # 266 models + keep_indices.npy
python scripts/generate_logits.py --config "$EXP/logit_config.yaml"         # per-model logits (LMDB)
python scripts/attacks.py        --config "$EXP/attack_config.yaml"         # LiRA/RMIA/Attack-R/LOSS scores
```

Membership masks use per-sample inclusion probability 1/2 (balanced across the
pool). Models train for 100 epochs with SGD (momentum 0.9, weight decay
5e-4, batch size 256, cosine schedule, initial LR 0.1) with random crops and
horizontal flips.

### 2b. RARE-side artifacts (per benchmark)

```bash
# Retrain the 10 targets, recording per-epoch loss trajectories.
python scripts/train.py --config "$EXP/train_config.yaml" --trace-mode full --overwrite

# One complement-trained reference per target (+ cached trajectory).
python scripts/rare_train_references.py --config configs/config_rare.yaml \
    --experiment-dir "$EXP" --num-references 1 --reference-seeds 42 --device cuda

# LT-MLP baseline scores (leave-one-out; see the script docstring for the
# full 11-invocation disjoint cover of all 10 targets).
python scripts/score_liu_mlp.py --experiment-dir "$EXP" --target-models 0 25
# ... (remaining pools)

# Target-model quality metrics (Appendix model-quality table).
python scripts/eval_final_models.py --config "$EXP/train_config.yaml"
```

RARE, LT-IQR, and LOSS require no scoring step; they are computed on the fly
from the trajectories and attack artifacts by the table/figure scripts below.

## 3. Regenerating tables and figures

Each script reads the on-disk artifacts and writes to
`paper_tabs_figs/figures/` (LaTeX + CSV) or `results/`.

| Paper element | Script |
|---|---|
| Table 1 — Main results | `paper_tabs_figs/tab_main.py` |
| Table 2 — Union budget sweep | `paper_tabs_figs/tab_union.py` |
| Table 3 — Cross-architecture | `paper_tabs_figs/tab_cross_arch.py` |
| Table 4 — Reference-statistic & component ablation | `paper_tabs_figs/tab_abl_final_vs_traj.py`, `abl_ref_last.py`, `abl_ref_minmax.py`, `tab_abl_components.py` |
| Table 5 — Budget equivalence | `scripts/budget_match_sweep.py` |
| Table 6 — Bottom-q window | `paper_tabs_figs/tab_abl_quantile.py` |
| Table 7 — Reference budget M | `paper_tabs_figs/tab_abl_refM.py` |
| Table 8 — Agreement-based gold | `analysis/agreement_gold.py` |
| Table 9 — Unsupervised selection | `paper_tabs_figs/tab_actionable_selection.py` |
| Table 10 — RARE-guided mitigation | `paper_tabs_figs/tab_defense.py` |
| Table 11 — Pipeline runtime | `paper_tabs_figs/tab_pipeline_cost.py` |
| Appendix — Target-model quality | `paper_tabs_figs/tab_model_quality.py` |
| Figure 1 — Top-3% overlap | `paper_tabs_figs/fig_teaser_overlap_top3.py` |
| Figure 2 — Qualitative examples | `paper_tabs_figs/fig_qualitative_examples.py` |
| Figure 3 — P@5% vs budget M | `paper_tabs_figs/fig_pareto.py` |
| Figure 4 — P@5% vs FPR | `paper_tabs_figs/fig_p5_vs_fpr.py` |
| Figure 5 — P@k% vs depth | `paper_tabs_figs/fig_p_vs_k.py` |
| Figure 6 — Ranking RBO | `paper_tabs_figs/fig_ranking_rbo.py` |

```bash
python paper_tabs_figs/tab_main.py     # -> paper_tabs_figs/figures/tab_main.tex
python paper_tabs_figs/fig_p_vs_k.py   # -> paper_tabs_figs/figures/fig_p_vs_k.pdf
```

## 4. Reproducibility notes

**Idempotency.** The RARE-side scripts skip completed work unless forced:

| Script | Skips when | Force with |
|---|---|---|
| `train.py --trace-mode full` | `best_model.pth` + `per_epoch_*.npy` present | `--overwrite` |
| `rare_train_references.py` | reference trajectory + checkpoint present | `--no-load-cache` |
| `score_liu_mlp.py` | `liu_mlp_scores_R*.npy` present | `--overwrite` |
| table/figure scripts | — (always cheap to re-run from saved scores) | n/a |

**Seeds.** Shadow/target training uses `seed = train_config.seed + index + 1`;
combined with the shared `init_model.pth`, retraining a target reproduces its
original trajectory, keeping `best_model.pth` consistent with the cached attack
scores. RARE references use `reference_seeds: [42]` (M=1). Seeds are logged
under each experiment directory.
