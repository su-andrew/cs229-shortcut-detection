import math
from pathlib import Path

import pandas as pd
import pytest

from src.saliency import (
    compute_label_aurocs,
    error_cases_to_frame,
    format_review_summary,
    gradcam_filename,
    lowest_auroc_labels,
    review_template_from_cases,
    select_error_cases,
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
