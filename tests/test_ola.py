import numpy as np

from src.ola import bootstrap_mean_ci, stratify_oll


def test_bootstrap_mean_ci_resamples_image_rows_deterministically():
    values = np.linspace(0.0, 1.0, 50)
    mean, lo, hi = bootstrap_mean_ci(values, n_boot=500, seed=229)
    assert abs(mean - values.mean()) < 1e-9
    assert lo < mean < hi
    mean2, lo2, hi2 = bootstrap_mean_ci(values, n_boot=500, seed=229)
    assert (mean, lo, hi) == (mean2, lo2, hi2)


def test_stratify_oll_splits_by_truth_and_error_type():
    rows = [
        {"y_true": 1, "y_pred": 0.9, "oll": 0.2},  # TP
        {"y_true": 1, "y_pred": 0.1, "oll": 0.8},  # FN
        {"y_true": 0, "y_pred": 0.9, "oll": 0.7},  # FP
        {"y_true": 0, "y_pred": 0.1, "oll": 0.3},  # TN
    ]
    strata = stratify_oll(rows, threshold=0.5)
    assert sorted(strata.keys()) == ["all", "fn", "fp", "y0", "y1"]
    assert strata["y1"] == [0.2, 0.8]
    assert strata["y0"] == [0.7, 0.3]
    assert strata["fp"] == [0.7]
    assert strata["fn"] == [0.8]
