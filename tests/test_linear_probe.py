import numpy as np
import pytest

from src.linear_probe import (
    global_average_pool,
    fit_probe,
    evaluate_probe,
)


def test_global_average_pool_shape_and_values():
    # (N, C, H, W) -> (N, C), mean over spatial dims
    feat = np.zeros((2, 3, 4, 4), dtype="float32")
    feat[0, 0] = 1.0   # channel-0 of sample-0 all ones -> mean 1
    feat[1, 2] = 2.0   # channel-2 of sample-1 all twos -> mean 2
    out = global_average_pool(feat)
    assert out.shape == (2, 3)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(0.0)
    assert out[1, 2] == pytest.approx(2.0)


def test_fit_and_evaluate_probe_separable():
    # linearly separable: positives at +1, negatives at -1 in a 4-d space
    rng = np.random.default_rng(229)
    Xpos = rng.normal(1.0, 0.05, size=(40, 4))
    Xneg = rng.normal(-1.0, 0.05, size=(40, 4))
    Xtr = np.vstack([Xpos, Xneg])
    ytr = np.array([1] * 40 + [0] * 40)
    probe = fit_probe(Xtr, ytr)
    assert probe is not None

    Xval = np.vstack([rng.normal(1.0, 0.05, size=(10, 4)),
                      rng.normal(-1.0, 0.05, size=(10, 4))])
    yval = np.array([1] * 10 + [0] * 10)
    auroc = evaluate_probe(probe, Xval, yval)
    assert auroc > 0.95  # near-perfect on a separable set


def test_fit_probe_skips_single_class():
    X = np.random.default_rng(0).normal(size=(20, 4))
    y = np.ones(20, dtype=int)  # only one class
    assert fit_probe(X, y) is None


def test_evaluate_probe_nan_on_single_class_val():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 4)); y = np.array([1] * 10 + [0] * 10)
    probe = fit_probe(X, y)
    # val set with one class -> AUROC undefined -> NaN, not a crash
    Xval = rng.normal(size=(5, 4)); yval = np.ones(5, dtype=int)
    assert np.isnan(evaluate_probe(probe, Xval, yval))
