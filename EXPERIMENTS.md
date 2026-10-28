# Reproduction Guide

How to reproduce every table, figure and in-text number of the paper from this
repository. Each paper element is regenerated from on-disk artifacts by one
script; Sections 2–3 describe how to produce those artifacts.

Run everything from the repository root. Analysis scripts are CPU-only; pin
`OMP_NUM_THREADS=1` so the K-means selection is bit-reproducible.

## 1. Environment and data

```bash
conda create -n rare python=3.10 && conda activate rare
pip install -e .
```

CIFAR10/100 download automatically to `data/`. For CINIC10, place the official
`train/`, `valid/`, `test/` folders under `data/cinic10/` and run:

```bash
python scripts/build_lmdb_cinic.py
```

CINIC10 uses a fixed random subsample of 50,000 training images
(`dataset.n_data_points` in its config); the indices are written to
`<exp>/subsample_indices.npy` on first use.

## 2. Benchmarks and configs

Each benchmark has its own config folder; `experiment.checkpoint_dir` in each
config is the experiment directory the scripts read and write.

| Benchmark | Config folder | Experiment directory |
|---|---|---|
| CIFAR10 / WRN-28-2 | `configs/cifar10_wrn28-2/` | `experiments/cifar10/wrn28-2/2025-12-07_0140` |
| CIFAR100 / WRN-28-2 | `configs/cifar100_wrn28-2/` | `experiments/cifar100/wrn28-2/2025-12-25_0121` |
| CINIC10 / WRN-28-2 | `configs/cinic10_wrn28-2/` | `experiments/cinic10/wrn28-2/2026-05-11_0215` |
| CIFAR10 / ResNet-20 (cross-arch.) | `configs/cifar10_resnet20/` | `experiments/cifar10/resnet20/2026-01-23_2258` |
| CIFAR10 / WRN-40-4 (cross-arch.) | `configs/cifar10_wrn40-4/` | `experiments/cifar10/wrn40-4/2026-01-24_2029` |
| CIFAR10 agreement gold (Table 10) | `configs/cifar10_wrn28-2_agreement/` | `experiments/cifar10/wrn28-2/agreement_2026-07-18` |

All benchmarks use the ten targets `[0, 25, 50, 75, 100, 125, 150, 175, 200, 250]`
(`target_models` in the configs). `configs/config_rare.yaml` holds the RARE
reference settings (M=1, seed 42).

## 3. Pipeline (per benchmark)

```bash
C=configs/cifar10_wrn28-2
EXP=experiments/cifar10/wrn28-2/2025-12-07_0140
```

### 3a. Shadow pool and reference attacks (one-time, compute-heavy)

```bash
python scripts/train.py           --config $C/train_config.yaml   # 266 models + keep_indices.npy
python scripts/generate_logits.py --config $C/logit_config.yaml   # per-model logits
python scripts/attacks.py         --config $C/attack_config.yaml  # LiRA / RMIA / Attack-R / LOSS
```

Membership uses per-sample inclusion probability 1/2. Models train for 100
epochs with SGD (momentum 0.9, weight decay 5e-4, batch size 256, cosine
schedule from LR 0.1), random 4-pixel-padded crops and horizontal flips;
attack logits come from the epoch-100 checkpoints. LiRA is online, averages
the original and flipped view, and uses the global-variance variant for
M < 64 shadows. The 256-shadow attacks define the gold sets.

### 3b. RARE-side artifacts

```bash
# Retrain the 10 targets, recording per-epoch loss (and margin) trajectories.
python scripts/train.py --config $C/train_config.yaml --trace-mode full --overwrite

# One complement-trained reference per target, with its per-epoch trajectory
# and (for Table 7) its logits.
python scripts/rare_train_references.py --config configs/config_rare.yaml \
    --experiment-dir $EXP --num-references 1 --reference-seeds 42 --device cuda

# LT-IQR scores (read by the tables and figures).
python scripts/score_lt_iqr.py --experiment-dir $EXP

# LT-MLP at M=1: five disjoint target pairs (see the script docstring).
python scripts/score_liu_mlp.py --experiment-dir $EXP --target-models 0 25
# ... 50 75 / 100 125 / 150 175 / 200 250

# Target-model performance (Table 8).
python scripts/eval_final_models.py --config $C/train_config.yaml
```

RARE itself needs no scoring step: the table and figure scripts compute
`z_T(LT-IQR) + z_T(LT-BTM-R)` from the trajectories.

**Multi-reference runs (CIFAR10 / WRN-28-2 only).** Tables 2, 6 and 12 use up
to eight references per target:

```bash
python scripts/rare_train_references.py --config configs/config_rare.yaml \
    --experiment-dir $EXP --num-references 8 \
    --reference-seeds 42 43 44 45 46 47 48 49 --device cuda
```

References 1–7 start from fresh random initializations and, like
reference 0, train on all of the target's non-members.

**Reference logits for Table 7.** `rare_train_references.py` caches the
reference logits by default; `scripts/save_reference_logits.py` regenerates
them for existing references.

### 3c. Agreement-based gold sets (Table 10)

```bash
C=configs/cifar10_wrn28-2_agreement
python analysis/agreement_gold.py --stage split                 # fixed split for 10 targets
python scripts/train.py --config $C/train_config.yaml --trace-mode full
python scripts/rare_train_references.py --config configs/config_rare.yaml \
    --experiment-dir experiments/cifar10/wrn28-2/agreement_2026-07-18 \
    --target-models 0 1 2 3 4 5 6 7 8 9 --num-references 1 --reference-seeds 42
python scripts/generate_logits.py --config $C/logit_config.yaml
python analysis/agreement_gold.py --stage score                 # LiRA-256 / LiRA-8 vs the CIFAR10 pool
python analysis/agreement_gold.py --stage analyze               # Table 10
```

## 4. Paper elements

Tables and figures are written to `paper_tabs_figs/tables/` and
`paper_tabs_figs/figures/`; each `.tex` file is the exact LaTeX of the paper's
table or figure environment (figures include `figures/<name>.pdf`). Scripts
also print the numbers quoted in the surrounding text.

| Paper element | Label | Script |
|---|---|---|
| Table 1 — Main results | `tab:main` | `paper_tabs_figs/tab_main.py` |
| Table 2 — Union gold sweep (CIFAR10) | `tab:union-cifar10` | `paper_tabs_figs/tab_union.py` |
| Table 3 — Cross-architecture | `tab:cross-arch` | `paper_tabs_figs/tab_cross_arch.py` |
| Table 4 — Pipeline runtime | `tab:pipeline-cost` | `paper_tabs_figs/tab_pipeline_cost.py` |
| Table 5 — Reference-statistic ablation | `tab:abl-final-vs-traj` | `paper_tabs_figs/tab_abl_final_vs_traj.py` |
| Table 6 — Budget equivalence | `tab:crossover` | `paper_tabs_figs/tab_crossover.py` |
| Table 7 — RARE-guided mitigation | `tab:defense` | `paper_tabs_figs/tab_defense.py` |
| Table 8 — Target-model performance | `tab:model-performance` | `paper_tabs_figs/tab_model_performance.py` |
| Table 9 — Precision / recall at fixed depths | `tab:precision-recall` | `paper_tabs_figs/tab_precision_recall.py` |
| Table 10 — Agreement-based gold sets | `tab:agreement-gold` | `analysis/agreement_gold.py --stage analyze` |
| Table 11 — Reference-side quantile q | `tab:abl-quantile` | `paper_tabs_figs/tab_abl_quantile.py` |
| Table 12 — Reference budget M | `tab:abl-refM` | `paper_tabs_figs/tab_abl_refM.py` |
| Figure 1 — Qualitative examples | `fig:qualitative_examples` | `paper_tabs_figs/fig_qualitative_examples.py` |
| Figure 2 — Ranking RBO | `fig:ranking-rbo` | `paper_tabs_figs/fig_ranking_rbo.py` |
| Figure 3 — P@5% vs reference budget | `fig:pareto` | `paper_tabs_figs/fig_pareto.py` |
| Figure 4 — P@5% vs gold-set FPR | `fig:p5_vs_fpr` | `paper_tabs_figs/fig_p5_vs_fpr.py` |
| Figure 5 — P@k% vs depth | `fig:p_vs_k` | `paper_tabs_figs/fig_p_vs_k.py` |
| Figure 6 — Top-3% overlap | `fig:teaser_overlap` | `paper_tabs_figs/fig_teaser_overlap_top3.py` |
| Text — early-stopped calibration (Sec. on Table 5) | — | `analysis/calib_es.py` |
| Text — unsupervised vulnerable-set selection | — | `analysis/unsupervised_selection.py` |

Tables 8–12 and Figures 4–6 are in the supplementary appendix
(`appendix.tex`), which continues the paper's numbering.

```bash
python paper_tabs_figs/tab_main.py      # -> paper_tabs_figs/tables/tab_main.tex
python paper_tabs_figs/fig_p_vs_k.py    # -> paper_tabs_figs/figures/fig_p_vs_k.{pdf,tex}
```

## 5. Reproducibility notes

**Idempotency.** The RARE-side scripts skip completed work unless forced:

| Script | Skips when | Force with |
|---|---|---|
| `train.py --trace-mode full` | `best_model.pth` + `per_epoch_*.npy` present | `--overwrite` |
| `rare_train_references.py` | reference trajectory + checkpoint present | `--no-load-cache` |
| `score_liu_mlp.py` | `liu_mlp_scores_R*.npy` present | `--overwrite` |
| table / figure scripts | — (cheap to re-run from saved scores) | n/a |

**Seeds.** Shadow and target training uses `seed = config.seed + index + 1`;
with the shared `init_model.pth`, retraining a target reproduces its original
trajectory, keeping `best_model.pth` consistent with the cached attack scores.
The M=1 reference uses seed 42.

**Runtime.** Table 4, the speedup columns of Tables 1 and 2, and the
runtime column of Table 7 are built from per-target timings:

* training-time phases (shadow, target and reference training, trajectory
  recording, LT-IQR and LT-MLP scoring) are appended to `<exp>/runtime.csv`
  by the pipeline scripts as they run;
* inference timings (per-model logit generation, cold attack scoring, the
  LOSS forward pass, and RARE score computation) are measured by

  ```bash
  python scripts/measure_runtime.py --config configs/<benchmark>/attack_config.yaml --stage all
  ```

  which writes to `<exp>/runtime_measurements/` and never overwrites the
  shipped timings. `python paper_tabs_figs/tab_pipeline_cost.py --timings-dir
  runtime_measurements` builds the table from the new measurements.

The paper's numbers use the timings shipped in each experiment directory,
measured on one NVIDIA RTX PRO 6000 Blackwell GPU; table speedups take, for
each training phase, the fastest complete logged run per model, so runs slowed
by concurrent GPU load do not inflate costs.
