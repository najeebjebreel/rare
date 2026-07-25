"""ROC-curve plotting helpers for attack results."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple


def plot_combined_roc(
    attack_results: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: str,
    log_scale: bool = True,
):
    """
    Plot combined ROC curves for all attacks on a single figure.
    Mirrors the official LiRA implementation style.

    Args:
        attack_results: {attack_name: (fpr, tpr, auc)}
        save_path: Path to save the figure
        log_scale: Use log scale for both axes
    """
    plt.figure(figsize=(4, 3))

    for attack_name, (fpr, tpr, roc_auc) in attack_results.items():
        plt.plot(fpr, tpr, label=f"{attack_name} (AUC={roc_auc * 100:.2f})%")

    if log_scale:
        plt.semilogx()
        plt.semilogy()
        plt.xlim(1e-5, 1)
        plt.ylim(1e-5, 1)
    else:
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(save_path, dpi=1000)
    plt.clf()
    plt.close()
