"""Cache reference-model logits for the RARE-guided logit-replacement defense.

The RARE reference trace (`ref_losses_*`) is a single-view, bf16, train-mode
eval and is NOT a valid basis for blending against the attack `logits.npy`.
This script (re)generates each reference model's final-checkpoint logits through
the SAME `LogitsGenerator` that produced the target/shadow `logits.npy`
(A=2 by default, the experiment's `inference_mode`, fp32) and saves
`ref_logits_E{E}_{tag}_ref{r}.npy` next to `ref_losses_*`. One forward per
reference; no retraining. Run once as a RARE pipeline step; the defense then
reads the logits off disk with zero recomputation.

Usage:
  python scripts/save_reference_logits.py --config experiments/<bench>/<run>/logit_config.yaml
  python scripts/save_reference_logits.py --config <logit_config.yaml> --num-refs 2 --overwrite
"""

import _bootstrap  # noqa: F401

import argparse
import logging
import sys
from pathlib import Path

import yaml

from rare.analysis.reference import generate_reference_logits

DEFAULT_TARGETS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=str,
                    help="Path to the experiment's logit_config.yaml (canonical "
                         "logit-generation basis).")
    ap.add_argument("--targets", type=int, nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--num-refs", type=int, default=1)
    ap.add_argument("--nb-augmentation", type=int, default=2,
                    help="2 (LiRA basis; view 0 also serves RMIA/Attack-R).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("save_reference_logits")

    cfg = yaml.safe_load(open(args.config))
    exp = cfg.get("experiment", {}).get("checkpoint_dir")
    if exp is None:
        print("logit config missing experiment.checkpoint_dir", file=sys.stderr)
        sys.exit(1)
    logger.info(f"experiment_dir = {exp}")
    logger.info(f"targets = {args.targets}  num_refs = {args.num_refs}  "
                f"A = {args.nb_augmentation}")

    generate_reference_logits(
        experiment_dir=str(exp),
        targets=list(args.targets),
        logit_config=cfg,
        num_refs=int(args.num_refs),
        logger=logger,
        overwrite=bool(args.overwrite),
        nb_augmentation=int(args.nb_augmentation),
    )
    logger.info("done")


if __name__ == "__main__":
    main()
