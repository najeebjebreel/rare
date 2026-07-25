"""
Run inference on every shadow model to produce logits for downstream MIAs.

Reads `train_config.yaml` saved by `train.py` from the experiment directory,
overlays the logits-stage config, and writes per-model `logits/logits.npy`.
"""

import _bootstrap  # noqa: F401

import argparse
import logging
import os
import sys

import yaml

import time

from rare.utils.common import (
    append_runtime,
    parse_overrides,
    recursive_update,
    save_config,
    set_seed,
    setup_logger,
    validate_config,
)
from rare.utils.logits import LogitsGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on member and non-member data to generate logits for MIAs'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument(
        '--override',
        nargs='*',
        default=[],
        help='Override config values. Format: key1.subkey=value key2=value'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        logit_config = yaml.safe_load(f)

    overrides = parse_overrides(args.override)
    recursive_update(logit_config, overrides)

    experiment_dir = logit_config.get('experiment', {}).get('checkpoint_dir', 'none')
    train_config_path = os.path.join(experiment_dir, 'train_config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f) or {}

    # Merge: train config as base, logits config + overrides on top.
    config = train_config
    recursive_update(config, logit_config)

    # If a flat `overwrite` is passed, normalize it into logits_generation.overwrite.
    logits_cfg = config.setdefault('logits_generation', {})
    if 'overwrite' in config and 'overwrite' not in logits_cfg:
        logits_cfg['overwrite'] = config['overwrite']

    required_keys = ['experiment.checkpoint_dir']
    try:
        validate_config(config, required_keys)
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)

    experiment_dir = config.get('experiment', {}).get('checkpoint_dir', 'none')

    log_level_str = config.get('experiment', {}).get('log_level', 'info').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger('Logit generation', os.path.join(experiment_dir, 'logit_log.log'), level=log_level)

    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    save_config(config, os.path.join(experiment_dir, 'logit_config.yaml'))
    logger.info(f"Saved configuration to {os.path.join(experiment_dir, 'logit_config.yaml')}")

    compute_logits = config.get('compute_logits', True)
    if compute_logits:
        logger.info("Generating logits ...")
        t0 = time.perf_counter()
        logits_generator = LogitsGenerator(config, logger)
        logits_generator.generate_logits()
        elapsed = time.perf_counter() - t0
        ds_cfg_log = config.get('dataset', {})
        append_runtime(
            os.path.join(experiment_dir, 'runtime.csv'),
            phase='generate_logits', item='all_models', seconds=elapsed,
            dataset=str(ds_cfg_log.get('name', '')).lower(),
            arch=config.get('model', {}).get('architecture', ''),
            loading=str(ds_cfg_log.get('loading', 'gpu')).lower(),
            n_data_points=('full' if ds_cfg_log.get('n_data_points') is None
                           else int(ds_cfg_log.get('n_data_points'))),
        )

    logger.info("Logits completed successfully")


if __name__ == '__main__':
    main()
