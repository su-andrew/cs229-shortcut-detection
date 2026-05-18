import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.saliency as saliency
from src.saliency import (
    compute_label_aurocs,
    error_cases_to_frame,
    format_review_summary,
    gradcam_filename,
    lowest_auroc_labels,
    model_target_label,
    render_gradcam,
    review_template_from_cases,
    resolve_image_path,
    select_error_cases,
    load_xray_tensor,
    summarize_review_annotations,
    validate_prediction_columns,
)


def test_validate_prediction_columns_reports_missing_contract() -> None:
    predictions = pd.DataFrame({"path": ["a.png"], "Edema_true": [1]})

    with pytest.raises(ValueError, match="Edema_pred"):
        validate_prediction_columns(predictions, ["Edema"])


def test_compute_label_aurocs_sorts_lowest_first() -> None:
    predictions = pd.DataFrame(
        {
            "path": ["a.png", "b.png", "c.png", "d.png"],
            "Edema_true": [0, 0, 1, 1],
            "Edema_pred": [0.8, 0.7, 0.3, 0.2],
            "Cardiomegaly_true": [0, 0, 1, 1],
            "Cardiomegaly_pred": [0.1, 0.2, 0.8, 0.9],
        }
    )

    aurocs = compute_label_aurocs(predictions, ["Edema", "Cardiomegaly"])

    assert math.isclose(aurocs["Edema"], 0.0)
    assert math.isclose(aurocs["Cardiomegaly"], 1.0)
    assert lowest_auroc_labels(aurocs, 1) == ["Edema"]


def test_compute_label_aurocs_excludes_uncertain_labels() -> None:
    # True regression test for both halves of the exclusion guard
    # (np.isin(y_true,(0,1)) & np.isfinite(y_pred)). The clean pair
    # (true=0 @ 0.4, true=1 @ 0.6) has AUROC exactly 1.0.
    #
    # Label-side: a CheXpert uncertain row (true=-1) with a *low* score
    # (0.1) is neither positive nor negative, so it does not change
    # n_pos/n_neg -- but if left in the ranking it pushes the positive
    # row's rank from 2 to 3, giving (3 - 1) / (1 * 1) = 2.0. Without the
    # isin guard this fixture yields AUROC 2.0; with it, exactly 1.0.
    #
    # Prediction-side: bad_pred.png is a clean negative (true=0) whose
    # *prediction* is NaN. If np.isfinite(y_pred) did not drop it, the
    # NaN would corrupt argsort ranking and the result would no longer
    # equal the clean AUROC. This row specifically exercises the
    # isfinite(y_pred) half of the guard.
    clean = pd.DataFrame(
        {
            "path": ["neg.png", "pos.png"],
            "Edema_true": [0, 1],
            "Edema_pred": [0.4, 0.6],
        }
    )
    with_uncertain = pd.concat(
        [
            clean,
            pd.DataFrame(
                {
                    "path": ["uncertain.png", "not_mentioned.png", "bad_pred.png"],
                    "Edema_true": [-1, np.nan, 0],
                    "Edema_pred": [0.1, 0.5, np.nan],
                }
            ),
        ],
        ignore_index=True,
    )

    clean_auroc = compute_label_aurocs(clean, ["Edema"])["Edema"]
    guarded_auroc = compute_label_aurocs(with_uncertain, ["Edema"])["Edema"]

    assert math.isclose(clean_auroc, 1.0)
    assert guarded_auroc <= 1.0
    assert math.isclose(guarded_auroc, clean_auroc)


def test_compute_label_aurocs_degenerate_splits_return_nan() -> None:
    # AUROC is undefined without both classes present after the {0,1} +
    # finite-pred filter. All-positive, all-negative, and empty-valid
    # (only -1/NaN labels) splits must each yield NaN, not an error or a
    # spurious number -- this is what makes Atelectasis (n_neg=0 under
    # impression_fixed) surface as NaN rather than crash the run.
    all_positive = pd.DataFrame(
        {"path": ["a.png", "b.png"], "Edema_true": [1, 1], "Edema_pred": [0.2, 0.9]}
    )
    all_negative = pd.DataFrame(
        {"path": ["a.png", "b.png"], "Edema_true": [0, 0], "Edema_pred": [0.2, 0.9]}
    )
    empty_valid = pd.DataFrame(
        {
            "path": ["a.png", "b.png"],
            "Edema_true": [-1, np.nan],
            "Edema_pred": [0.2, 0.9],
        }
    )

    assert math.isnan(compute_label_aurocs(all_positive, ["Edema"])["Edema"])
    assert math.isnan(compute_label_aurocs(all_negative, ["Edema"])["Edema"])
    assert math.isnan(compute_label_aurocs(empty_valid, ["Edema"])["Edema"])


# Note: select_error_cases applies the same explicit isin((0,1)) filter for
# contract consistency with baseline.py / compute_label_aurocs, but its
# false-positive/false-negative predicates (y_true == 0 / y_true == 1)
# already exclude -1 rows structurally. No fixture can make that function's
# output change when the explicit guard is removed, so there is no separate
# regression test for it -- the guard is defensive documentation, not a
# behavior change. The AUROC test above is the real regression guard.


def test_select_error_cases_picks_confident_fp_and_fn() -> None:
    predictions = pd.DataFrame(
        {
            "path": ["fp1.png", "fp2.png", "fn1.png", "fn2.png", "ok.png"],
            "Edema_true": [0, 0, 1, 1, 1],
            "Edema_pred": [0.95, 0.6, 0.05, 0.4, 0.8],
        }
    )

    cases = select_error_cases(predictions, ["Edema"], top_k_per_type=1)
    selected = error_cases_to_frame(cases)

    assert selected["path"].tolist() == ["fp1.png", "fn1.png"]
    assert selected["error_type"].tolist() == ["false_positive", "false_negative"]
    assert selected["confidence"].tolist() == [0.95, 0.95]


def test_review_template_points_to_expected_gradcam_paths() -> None:
    predictions = pd.DataFrame(
        {
            "path": ["fp1.png"],
            "Pleural Effusion_true": [0],
            "Pleural Effusion_pred": [0.95],
        }
    )

    case = select_error_cases(predictions, ["Pleural Effusion"], top_k_per_type=1)[0]
    review = review_template_from_cases(
        [case], Path("results/figures"), max_figures=1
    )

    assert gradcam_filename(case) == "pleural_effusion_false_positive_1.png"
    assert review.loc[0, "gradcam_path"] == (
        "results/figures/pleural_effusion_false_positive_1.png"
    )
    assert review.loc[0, "rendered_by_default"]
    assert review.loc[0, "attention_location"] == ""
    assert review.loc[0, "shortcut_category"] == ""


def test_resolve_image_path_preserves_existing_relative_path(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "data" / "chexpert" / "PNG_valid" / "case.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not a real image")
    monkeypatch.chdir(tmp_path)

    resolved = resolve_image_path(
        Path("data/chexpert"), "data/chexpert/PNG_valid/case.png"
    )

    assert resolved == Path("data/chexpert/PNG_valid/case.png")


def test_model_target_label_maps_pleural_effusion_to_xrv_effusion() -> None:
    assert model_target_label(
        "Pleural Effusion", ["Atelectasis", "Effusion"]
    ) == "Effusion"


def test_load_xray_tensor_uses_shared_data_loader(monkeypatch) -> None:
    def fake_load_xray_image(path):
        assert path == Path("case.dcm")
        return np.ones((1, 2, 2), dtype=np.float32)

    def fake_transform():
        return lambda image: image * 2

    monkeypatch.setattr(saliency, "load_xray_image", fake_load_xray_image)
    monkeypatch.setattr(saliency, "default_xrv_transform", fake_transform)

    tensor = load_xray_tensor(Path("case.dcm"))

    assert tuple(tensor.shape) == (1, 1, 2, 2)
    assert float(tensor.max()) == 2.0


def test_render_gradcam_selects_target_and_writes_overlay(tmp_path, monkeypatch) -> None:
    # render_gradcam is the highest-risk previously-untested path. Inject
    # fake matplotlib / pytorch_grad_cam modules and a fake model so the
    # whole path runs without torch: target-layer pick, label->xrv-index
    # mapping, Grad-CAM invocation, RGB normalization, and overlay write.
    import sys
    import types

    recorded: dict = {}

    mpl = types.ModuleType("matplotlib")
    plt_mod = types.ModuleType("matplotlib.pyplot")

    def fake_imsave(path, arr):
        recorded["imsave_path"] = path
        recorded["overlay_shape"] = np.asarray(arr).shape

    plt_mod.imsave = fake_imsave
    mpl.pyplot = plt_mod

    pgc = types.ModuleType("pytorch_grad_cam")

    class FakeGradCAM:
        def __init__(self, model, target_layers):
            recorded["target_layers"] = target_layers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __call__(self, input_tensor, targets):
            recorded["cam_targets"] = targets
            return np.ones((1, 4, 4), dtype=np.float32)

    pgc.GradCAM = FakeGradCAM

    pgc_utils = types.ModuleType("pytorch_grad_cam.utils")
    pgc_image = types.ModuleType("pytorch_grad_cam.utils.image")

    def fake_show_cam_on_image(rgb, cam, use_rgb=False):
        recorded["rgb_shape"] = rgb.shape
        recorded["use_rgb"] = use_rgb
        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    pgc_image.show_cam_on_image = fake_show_cam_on_image

    pgc_targets = types.ModuleType("pytorch_grad_cam.utils.model_targets")

    class FakeClassifierOutputTarget:
        def __init__(self, index):
            recorded["target_index"] = index

    pgc_targets.ClassifierOutputTarget = FakeClassifierOutputTarget

    for name, mod in {
        "matplotlib": mpl,
        "matplotlib.pyplot": plt_mod,
        "pytorch_grad_cam": pgc,
        "pytorch_grad_cam.utils": pgc_utils,
        "pytorch_grad_cam.utils.image": pgc_image,
        "pytorch_grad_cam.utils.model_targets": pgc_targets,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    class FakeTensor:
        def __init__(self, arr):
            self._a = arr

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._a

    image_tensor = FakeTensor(np.zeros((1, 1, 4, 4), dtype=np.float32))

    norm5_layer = object()

    class FakeFeatures:
        norm5 = norm5_layer

    class FakeModel:
        features = FakeFeatures()
        pathologies = ["Atelectasis", "Effusion"]

    out = tmp_path / "figs" / "case.png"
    render_gradcam(FakeModel(), image_tensor, "Pleural Effusion", out)

    # Pleural Effusion -> xrv "Effusion" -> index 1 in pathologies.
    assert recorded["target_index"] == 1
    assert recorded["target_layers"] == [norm5_layer]
    assert recorded["rgb_shape"] == (4, 4, 3)
    assert recorded["use_rgb"] is True
    assert recorded["imsave_path"] == out
    assert out.parent.exists()


def test_summarize_review_annotations_counts_shortcut_findings() -> None:
    review = pd.DataFrame(
        {
            "label": ["Edema", "Edema", "Cardiomegaly"],
            "gradcam_path": ["a.png", "b.png", "c.png"],
            "rendered_by_default": [True, True, False],
            "attention_location": ["outside_lung", "mixed", "inside_lung"],
            "shortcut_category": ["laterality_marker", "support_device", "none"],
            "include_in_report": ["yes", "", "true"],
            "notes": ["corner marker", "device cue", "looks plausible"],
        }
    )

    summary = summarize_review_annotations(review)
    summary_text = format_review_summary(summary)

    assert summary["reviewed"] == 3
    assert summary["outside_lung"] == 1
    assert summary["outside_or_mixed"] == 2
    assert summary["included_for_report"] == 2
    assert summary["shortcut_category_counts"] == {
        "laterality_marker": 1,
        "support_device": 1,
    }
    assert "2 showed peak or mixed attention outside the lung field" in summary_text
