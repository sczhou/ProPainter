"""
Dumps things to tensorboard and console
"""

import os
import logging
import datetime
from typing import Dict
import numpy as np
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from tracker.utils.time_estimator import TimeEstimator


def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np


def detach_to_cpu(x):
    return x.detach().cpu()


def fix_width_trunc(x):
    return ('{:.9s}'.format('{:0.9f}'.format(x)))


class TensorboardLogger:
    def __init__(self, run_dir, py_logger: logging.Logger, *, enabled_tb):
        self.run_dir = run_dir
        self.py_log = py_logger
        if enabled_tb:
            self.tb_log = SummaryWriter(run_dir)
        else:
            self.tb_log = None

        # Get current git info for logging
        try:
            import git
            repo = git.Repo(".")
            git_info = str(repo.active_branch) + ' ' + str(repo.head.commit.hexsha)
        except (ImportError, RuntimeError):
            print('Failed to fetch git info. Defaulting to None')
            git_info = 'None'

        self.log_string('git', git_info)

        # used when logging metrics
        self.time_estimator: TimeEstimator = None

    def log_scalar(self, tag, x, it):
        if self.tb_log is None:
            return
        self.tb_log.add_scalar(tag, x, it)

    def log_metrics(self, exp_id, prefix, metrics: Dict, it):
        msg = f'{exp_id}-{prefix} - it {it:6d}: '
        metrics_msg = ''
        for k, v in sorted(metrics.items()):
            self.log_scalar(f'{prefix}/{k}', v, it)
            metrics_msg += f'{k: >10}:{v:.7f},\t'

        if self.time_estimator is not None:
            self.time_estimator.update()
            avg_time = self.time_estimator.get_and_reset_avg_time()
            est = self.time_estimator.get_est_remaining(it)
            est = datetime.timedelta(seconds=est)
            if est.days > 0:
                remaining_str = f'{est.days}d {est.seconds // 3600}h'
            else:
                remaining_str = f'{est.seconds // 3600}h {(est.seconds%3600) // 60}m'
            eta = datetime.datetime.now() + est
            eta_str = eta.strftime('%Y-%m-%d %H:%M:%S')
            time_msg = f'avg_time:{avg_time:.3f},remaining:{remaining_str},eta:{eta_str},\t'
            msg = f'{msg} {time_msg}'

        msg = f'{msg} {metrics_msg}'
        self.py_log.info(msg)

    def log_image(self, stage_name, tag, image, it):
        image_dir = os.path.join(self.run_dir, f'{stage_name}_images')
        os.makedirs(image_dir, exist_ok=True)

        image = Image.fromarray(image)
        image.save(os.path.join(image_dir, f'{tag}_{it}.png'))

    def log_string(self, tag, x):
        self.py_log.info(f'{tag} - {x}')
        if self.tb_log is None:
            return
        self.tb_log.add_text(tag, x)

    def debug(self, x):
        self.py_log.debug(x)

    def info(self, x):
        self.py_log.info(x)

    def warning(self, x):
        self.py_log.warning(x)

    def error(self, x):
        self.py_log.error(x)

    def critical(self, x):
        self.py_log.critical(x)
