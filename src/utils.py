"""Shared utilities."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch if it is installed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ModuleNotFoundError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
