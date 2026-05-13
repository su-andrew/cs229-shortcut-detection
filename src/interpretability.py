"""Interpretability helpers for saliency audits."""

from __future__ import annotations

import numpy as np


def attribution_outside_mask(attribution: np.ndarray, mask: np.ndarray) -> float:
    """Return fraction of positive attribution outside a binary mask."""
    if attribution.shape != mask.shape:
        raise ValueError("attribution and mask must have the same shape")

    positive = np.clip(attribution.astype(float), a_min=0.0, a_max=None)
    total = positive.sum()
    if total == 0:
        return 0.0

    return float(positive[~mask.astype(bool)].sum() / total)
