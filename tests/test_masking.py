import numpy as np
import pandas as pd
import pytest

from src.masking import (
    make_shortcut_mask,
    apply_mask,
    delta_auroc,
    delta_auroc_ci,
    rank_sort,
)


def test_make_shortcut_mask_geometry():
    corners = make_shortcut_mask("corners", shape=(224, 224), frac=0.15)
    assert corners.dtype == bool and corners.shape == (224, 224)
    assert corners[0, 0] and corners[-1, -1]
    assert not corners[112, 112]
    border = make_shortcut_mask("border", shape=(224, 224), frac=0.1)
    assert border[0, :].all() and border[:, 0].all()
    assert not border[112, 112]
    lat = make_shortcut_mask("laterality", shape=(224, 224), frac=0.15)
    assert lat[0, 0] and lat[0, -1]
    assert not lat[-1, 112]


def test_make_shortcut_mask_rejects_unknown_and_oversized():
    with pytest.raises(ValueError):
        make_shortcut_mask("nonsense")
    with pytest.raises(ValueError):
        make_shortcut_mask("corners", frac=0.9)  # would mask whole image


def test_apply_mask_mean_fill_replaces_region():
    # non-uniform image so mean-fill is a real, detectable change
    plane = np.arange(224 * 224, dtype="float32").reshape(224, 224)
    img = plane[None, :, :].copy()  # (1,224,224)
    expected_mean = float(plane.mean())
    mask = make_shortcut_mask("corners", shape=(224, 224), frac=0.15)

    out = apply_mask(img, mask, fill="mean")
    assert out.dtype == np.float32
    assert np.allclose(out[0][mask], expected_mean)        # masked -> global mean
    assert np.allclose(out[0][~mask], plane[~mask])         # rest untouched
    assert np.array_equal(img, plane[None, :, :])           # input not mutated

    out0 = apply_mask(img, mask, fill="zero")
    assert np.allclose(out0[0][mask], 0.0)


def test_apply_mask_rejects_unknown_fill():
    img = np.ones((1, 4, 4), dtype="float32")
    mask = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError):
        apply_mask(img, mask, fill="median")


def test_delta_auroc_positive_when_masking_hurts():
    y_true = np.array([1, 1, 0, 0])
    clean = np.array([0.9, 0.8, 0.2, 0.1])
    masked = np.array([0.4, 0.6, 0.5, 0.55])
    d = delta_auroc(y_true, clean, masked)
    assert d > 0


def test_delta_auroc_ignores_nonfinite_predictions():
    # A NaN prediction row must be dropped, not crash roc_auc_score.
    y_true = np.array([1, 1, 0, 0])
    clean = np.array([0.9, 0.8, 0.2, 0.1])           # perfect ranking on valid rows
    masked = np.array([0.9, np.nan, 0.2, 0.1])        # one NaN -> drop that row
    d = delta_auroc(y_true, clean, masked)
    # surviving rows (drop the NaN) are perfectly ranked in both clean and
    # masked -> AUROC 1.0 each -> ΔAUROC exactly 0.0 (finite, not just non-NaN)
    assert d == pytest.approx(0.0)


def test_delta_auroc_ci_brackets_point_estimate_and_is_seeded():
    rng = np.random.default_rng(0)
    n = 80
    y = np.array([1] * 40 + [0] * 40)
    # clean separates well; masked degrades -> positive delta
    clean = np.concatenate([rng.uniform(0.6, 1.0, 40), rng.uniform(0.0, 0.4, 40)])
    masked = np.concatenate([rng.uniform(0.4, 0.8, 40), rng.uniform(0.2, 0.6, 40)])
    point = delta_auroc(y, clean, masked)
    lo, hi = delta_auroc_ci(y, clean, masked, n_boot=500, seed=229)
    assert lo <= hi
    assert lo <= point <= hi          # CI brackets the point estimate
    # determinism under fixed seed
    l2, h2 = delta_auroc_ci(y, clean, masked, n_boot=500, seed=229)
    assert (lo, hi) == (l2, h2)


def test_delta_auroc_ci_nan_when_unscoreable():
    # single-class -> CI undefined -> (nan, nan), no crash
    y = np.array([1, 1, 1, 1])
    lo, hi = delta_auroc_ci(y, np.array([.9, .8, .7, .6]),
                            np.array([.5, .4, .3, .2]), n_boot=100, seed=1)
    assert np.isnan(lo) and np.isnan(hi)


def test_rank_sort_keeps_ranked_above_unranked():
    df = pd.DataFrame([
        {"label": "lowDelta_ranked", "delta_auroc": 0.01, "ranked": True},
        {"label": "highDelta_unranked", "delta_auroc": 0.50, "ranked": False},
        {"label": "highDelta_ranked", "delta_auroc": 0.30, "ranked": True},
    ])
    order = list(rank_sort(df)["label"])
    # both ranked rows come first (sorted by ΔAUROC), unranked last despite its
    # larger ΔAUROC
    assert order == ["highDelta_ranked", "lowDelta_ranked", "highDelta_unranked"]
