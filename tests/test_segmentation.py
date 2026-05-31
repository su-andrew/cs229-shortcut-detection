import numpy as np
import pytest

from src.segmentation import binarize_lung_logits, resample_to_224, union_lung_mask


def test_resample_to_224_shape():
    arr = np.random.rand(512, 512).astype("float32")
    out = resample_to_224(arr)
    assert out.shape == (224, 224)
    assert out.dtype == np.float32


def test_binarize_lung_logits_is_bool_and_thresholds():
    logits = np.array([[-5.0, 5.0], [5.0, -5.0]], dtype="float32")  # sigmoid -> ~0/1
    mask = binarize_lung_logits(logits, threshold=0.5)
    assert mask.dtype == bool
    assert mask.tolist() == [[False, True], [True, False]]


def test_union_lung_mask_no_logit_cancellation():
    # Two lung channels: left lung active top-left, right lung active bottom-right.
    # Where one lung is strongly positive, the other is strongly negative.
    # Summing logits first would cancel to ~0 (=> dropped from mask); a correct
    # per-channel-sigmoid union must KEEP both lung regions.
    left = np.array([[6.0, -6.0], [-6.0, -6.0]], dtype="float32")
    right = np.array([[-6.0, -6.0], [-6.0, 6.0]], dtype="float32")
    stack = np.stack([left, right], axis=0)  # (2,2,2)

    mask = union_lung_mask(stack, threshold=0.5)
    assert mask.dtype == bool
    assert mask.tolist() == [[True, False], [False, True]]  # both lungs kept

    # Contrast: summing logits then thresholding would WRONGLY drop both
    # (6 + -6 = 0 -> sigmoid 0.5, not > 0.5), proving the bug is avoided.
    summed = binarize_lung_logits(stack.sum(axis=0), threshold=0.5)
    assert summed.tolist() == [[False, False], [False, False]]
