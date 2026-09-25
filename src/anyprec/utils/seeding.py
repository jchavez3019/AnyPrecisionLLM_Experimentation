"""Global seeding for reproducible runs (ADR 0002, Reproducibility)."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python's ``random``, NumPy's global generator, and PyTorch on the CPU and CUDA.

    :param seed: The run's top-level ``seed``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
