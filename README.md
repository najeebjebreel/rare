# RARE: Reference-Augmented Ranking by Epoch Statistics for Membership-Inference Vulnerability Assessment

Official implementation of *RARE* (under submission). RARE ranks a target
model's known training members by their vulnerability to membership-inference
attacks, using two per-sample loss-trajectory statistics — one from the target
and one from a **single** complement-trained reference model — with no large
shadow ensemble and no learned attack classifier:

```
RARE_i = z(LT-IQR_i) + z(LT-BTM-R_i)
```

where `z(·)` standardizes a statistic over the target-member set, `LT-IQR` is
the interquartile range of the target's per-epoch loss trajectory, and
`LT-BTM-R` is the mean of the lowest-loss epochs of the reference's trajectory
(out-of-training difficulty). Against a 256-model LiRA audit at
FPR = 10⁻⁴, RARE matches or exceeds low-budget reference attacks that use
8–16× more reference models.

## Installation

```bash
conda create -n rare python=3.10 && conda activate rare
pip install -e .            # or: pip install -r requirements.txt
```

Experiments require a CUDA GPU; results in the paper use a single NVIDIA RTX
PRO 6000 (Blackwell).

## Repository structure

```
rare/                 importable library
  analysis/utils.py   RARE / LT-IQR / LT-BTM-R scoring and P@k metrics
  attacks/            MIA signal helpers
  utils/              training and data plumbing
scripts/              pipeline entry points (train, references, baselines)
paper_tabs_figs/      one script per paper table/figure
  tables/             generated table LaTeX + CSV
  figures/            generated figure PDF + CSV
configs/              YAML configs for each stage
EXPERIMENTS.md        full run order and code-to-paper map
```

## Data

CIFAR-10 and CIFAR-100 download automatically to `data/` on first use. For
CINIC-10, place the official `train/`, `valid/`, `test/` folders under
`data/cinic10/` and build the LMDB store:

```bash
python scripts/build_lmdb_cinic.py
```

## Reproducing the paper

The gold standard (a 266-model shadow pool, its logits, and the LiRA / RMIA /
Attack-R / LOSS attacks) is a one-time, compute-heavy prerequisite produced by
`scripts/train.py`, `scripts/generate_logits.py`, and `scripts/attacks.py`;
see [EXPERIMENTS.md](EXPERIMENTS.md) for the exact commands. Given a benchmark
directory with that pool in place, the RARE-specific steps are:

```bash
EXP=experiments/cifar10/wrn28-2/2025-12-07_0140

# 1. Retrain the 10 target models, recording per-epoch loss trajectories.
python scripts/train.py --config "$EXP/train_config.yaml" --trace-mode full --overwrite

# 2. Train one complement-trained reference per target and cache its trajectory.
python scripts/rare_train_references.py --config configs/config_rare.yaml \
    --experiment-dir "$EXP" --num-references 1 --reference-seeds 42 --device cuda

# 3. Score the LT-MLP baseline (RARE, LT-IQR, and LOSS need no scoring step;
#    they are computed on the fly by the table scripts). See the
#    scripts/score_liu_mlp.py docstring for the full 11-invocation cover.
python scripts/score_liu_mlp.py --experiment-dir "$EXP" --target-models 0 25
```

Each paper table and figure is regenerated from the on-disk artifacts by a
single script, e.g.:

```bash
python paper_tabs_figs/tab_main.py           # Table 1  -> tables/tab_main.tex
python paper_tabs_figs/fig_p_vs_k.py         # Figure 5 -> figures/fig_p_vs_k.pdf
```

## Using RARE

Given a target's per-epoch loss trajectory and one reference trajectory
(`(N, E)` arrays of per-sample losses), the RARE score is:

```python
import numpy as np
from rare.analysis.utils import load_rare_locked_scores, tail_metrics_at_k

# Full RARE = z(LT-IQR) + z(LT-BTM-R) for one target in an experiment dir.
scores = load_rare_locked_scores(EXP, target_model=0, num_refs=1, q=0.5)

# Precision / Recall of the top-k% against a vulnerable (gold) set. The
# paper's convention retrieves ceil(k% * n_members) and does not truncate
# the budget at |V|:
metrics = tail_metrics_at_k(scores, members, [0.01, 0.03, 0.05],
                            vulnerable=vulnerable, gold_scores=gold,
                            cap_k_at_gold=False, k_rounding="ceil")
```

The low-level statistics `score_ltiqr_trajectory` and
`score_avg_min_q_reference_losses` are also exposed in `rare.analysis.utils`.

## Citation

```bibtex
@inproceedings{rare_anonymous,
  title  = {RARE: Reference-Augmented Ranking by Epoch Statistics for 
  Membership-Inference Vulnerability Assessment},
  author = {Anonymous},
  note   = {Under submission},
  year   = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`).
