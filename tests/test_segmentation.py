import numpy as np
import pytest

from src.segmentation import binarize_lung_logits, resample_to_224


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
