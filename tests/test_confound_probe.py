import numpy as np
import pandas as pd
import pytest

from src.confound_probe import label_correlation, cv_probe_auroc


def test_label_correlation_computes_per_disease_rates():
    meta = pd.DataFrame({
        "ap_pa": ["AP", "AP", "PA", "PA", "Lateral"],
        "Edema": [1.0, 1.0, 0.0, -1.0, 1.0],   # AP: 2 pos/2; PA: 0 pos / 1 scorable
    })
    out = label_correlation(meta, ["Edema"]).set_index("label").loc["Edema"]
    assert out["ap_pos_rate"] == pytest.approx(1.0)
    assert out["pa_pos_rate"] == pytest.approx(0.0)
    assert out["n_ap"] == 2 and out["n_pa"] == 1   # Lateral + uncertain dropped


def test_cv_probe_auroc_separable_is_high():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(1.0, 0.1, (30, 8)), rng.normal(-1.0, 0.1, (30, 8))])
    y = np.array([1] * 30 + [0] * 30)
    mean, lo, hi = cv_probe_auroc(X, y, n_splits=5, seed=229)
    assert mean > 0.95 and lo <= mean <= hi


def test_cv_probe_auroc_single_class_is_nan():
    X = np.random.default_rng(1).normal(size=(20, 8))
    y = np.ones(20, dtype=int)
    mean, lo, hi = cv_probe_auroc(X, y)
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_cv_probe_auroc_caps_folds_to_minority():
    # only 3 minority examples -> must not request 5 folds and crash
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(1, 0.1, (3, 6)), rng.normal(-1, 0.1, (30, 6))])
    y = np.array([1, 1, 1] + [0] * 30)
    mean, lo, hi = cv_probe_auroc(X, y, n_splits=5, seed=229)
    assert not np.isnan(mean)   # ran without error despite tiny minority class
