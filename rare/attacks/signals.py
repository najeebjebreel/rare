"""
Signal conversions from raw logits.

Adapted from the RMIA codebase:
https://github.com/privacytrustlab/ml_privacy_meter/blob/d32734161a3395211fe5f3cd461932290b1fafbe/research/2024_rmia/signals.py
"""

import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def factorial(n):
    fact = 1
    for i in range(2, n + 1):
        fact = fact * i
    return fact


def get_taylor(logit_signals, n):
    power = logit_signals
    taylor = power + 1.0
    for i in range(2, n):
        power = power * logit_signals
        taylor = taylor + (power / factorial(i))
    return taylor


def convert_signals(all_logits, all_true_labels, metric, temp, extra=None):
    if metric == 'softmax':
        logit_signals = torch.div(all_logits, temp)
        max_logit_signals, max_indices = torch.max(logit_signals, dim=1)
        logit_signals = torch.sub(logit_signals, max_logit_signals.reshape(-1, 1))
        exp_logit_signals = torch.exp(logit_signals)
        exp_logit_sum = exp_logit_signals.sum(axis=1).reshape(-1, 1)
        true_exp_logit = exp_logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        output_signals = torch.div(true_exp_logit, exp_logit_sum)
    elif metric == 'taylor':
        n = extra["taylor_n"]
        taylor_signals = get_taylor(all_logits, n)
        taylor_logit_sum = taylor_signals.sum(axis=1).reshape(-1, 1)
        true_taylor_logit = taylor_signals.gather(1, all_true_labels.reshape(-1, 1))
        output_signals = torch.div(true_taylor_logit, taylor_logit_sum)
    elif metric == 'soft-margin':
        m = float(extra["taylor_m"])
        logit_signals = torch.div(all_logits, temp)
        exp_logit_signals = torch.exp(logit_signals)
        exp_logit_sum = exp_logit_signals.sum(axis=1).reshape(-1, 1)
        true_logits = logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        exp_true_logit = exp_logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        exp_logit_sum = exp_logit_sum - exp_true_logit
        soft_true_logit = torch.exp(true_logits - m)
        exp_logit_sum = exp_logit_sum + soft_true_logit
        output_signals = torch.div(soft_true_logit, exp_logit_sum)
    elif metric == 'taylor-soft-margin':
        m, n = float(extra["taylor_m"]), int(extra["taylor_n"])
        logit_signals = torch.div(all_logits, temp)
        taylor_logits = get_taylor(logit_signals, n)
        taylor_logit_sum = taylor_logits.sum(axis=1).reshape(-1, 1)
        true_logit = logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        taylor_true_logit = taylor_logits.gather(1, all_true_labels.reshape(-1, 1))
        taylor_logit_sum = taylor_logit_sum - taylor_true_logit
        soft_taylor_true_logit = get_taylor(true_logit - m, n)
        taylor_logit_sum = taylor_logit_sum + soft_taylor_true_logit
        output_signals = torch.div(soft_taylor_true_logit, taylor_logit_sum)
    elif metric == 'logits':
        output_signals = all_logits
    elif metric == 'target_logit':
        # Raw target-class logit — matches the LT-IQR repo's
        # ModelConfidenceExtractor.get_target_logits (computationalprivacy/loss_traces).
        # No softmax, no temperature scaling.
        output_signals = all_logits.gather(1, all_true_labels.reshape(-1, 1))
    elif metric == 'log-logit-scaling':
        # LiRA logit signal from "Membership Inference Attacks From First Principles"
        # https://arxiv.org/abs/2112.03570
        # Adapted from https://github.com/carlini/privacy/blob/better-mi/research/mi_lira_2021/score.py
        predictions = all_logits - torch.max(all_logits, dim=1, keepdim=True).values
        predictions = torch.exp(predictions)
        predictions = predictions / torch.sum(predictions, dim=1, keepdim=True)
        COUNT = predictions.shape[0]
        y_true = predictions[np.arange(COUNT), all_true_labels[:COUNT]]
        predictions[np.arange(COUNT), all_true_labels[:COUNT]] = 0
        y_wrong = torch.sum(predictions, dim=1)
        output_signals = (torch.log(y_true + 1e-45) - torch.log(y_wrong + 1e-45))

    output_signals = torch.flatten(output_signals)
    return output_signals
