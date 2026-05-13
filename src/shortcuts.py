"""Shortcut categories and simple scoring helpers."""

from __future__ import annotations

import numpy as np


SHORTCUT_CATEGORIES = [
    "metadata_overlay",
    "laterality_marker",
    "support_device",
    "acquisition_artifact",
]


def shortcut_reliance_score(original_probs: np.ndarray, masked_probs: np.ndarray) -> float:
    """Average probability drop after masking suspected shortcut regions."""
    if original_probs.shape != masked_probs.shape:
        raise ValueError("original_probs and masked_probs must have the same shape")

    return float(np.mean(original_probs.astype(float) - masked_probs.astype(float)))
