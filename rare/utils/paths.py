"""Centralized path management for experiment directory layout."""

from pathlib import Path


class ExperimentPaths:
    """
    Wrapper around an experiment directory exposing the conventional layout:

        <experiment_dir>/
          keep_indices.npy
          model_<i>/
            best.pth (and friends)
            logits/logits.npy
            scores/scores.npy
          ...
    """

    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)

    @property
    def keep_indices_path(self):
        return self.experiment_dir / 'keep_indices.npy'

    @property
    def train_test_stats_path(self):
        return self.experiment_dir / 'train_test_stats.csv'

    @property
    def roc_curve_single_path(self):
        return self.experiment_dir / 'roc_curve_single.pdf'

    @property
    def attack_results_single_path(self):
        return self.experiment_dir / 'attack_results_single.csv'

    @property
    def attack_results_loo_summary_path(self):
        return self.experiment_dir / 'attack_results_leave_one_out_summary.csv'

    @property
    def membership_labels_path(self):
        return self.experiment_dir / 'membership_labels.npy'

    @property
    def threshold_info_loo_path(self):
        return self.experiment_dir / 'threshold_info_leave_one_out.csv'

    def get_model_dir(self, model_idx):
        return self.experiment_dir / f'model_{model_idx}'

    def get_checkpoint_path(self, model_idx, checkpoint_name='best.pth'):
        return self.get_model_dir(model_idx) / checkpoint_name

    def get_logits_dir(self, model_idx):
        return self.get_model_dir(model_idx) / 'logits'

    def get_logits_path(self, model_idx, filename='logits.npy'):
        return self.get_logits_dir(model_idx) / filename

    def get_scores_dir(self, model_idx):
        return self.get_model_dir(model_idx) / 'scores'

    def get_scores_path(self, model_idx, filename='scores.npy'):
        return self.get_scores_dir(model_idx) / filename

    def get_loo_scores_path(self, score_filename):
        return self.experiment_dir / f'{score_filename}_leave_one_out.npy'

    def ensure_dir(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_model_dirs(self, model_idx):
        self.ensure_dir(self.get_model_dir(model_idx))
        self.ensure_dir(self.get_logits_dir(model_idx))
        self.ensure_dir(self.get_scores_dir(model_idx))

    def __str__(self):
        return f"ExperimentPaths(experiment_dir={self.experiment_dir})"
