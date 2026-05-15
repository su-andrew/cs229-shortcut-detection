'''
Expected prediction columns:

```text
path
Atelectasis_true, Atelectasis_pred
Cardiomegaly_true, Cardiomegaly_pred
Consolidation_true, Consolidation_pred
Edema_true, Edema_pred
Pleural Effusion_true, Pleural Effusion_pred
```

Select the lowest-AUROC labels and high-confidence FP/FN cases without rendering:

```bash
python -m src.saliency --predictions-csv results/predictions_val.csv --select-only
```

Render the first Grad-CAM overlays once images are present:

```bash
python -m src.saliency --predictions-csv results/predictions_val.csv --image-root data/chexpert

'''

import math

import pandas as pd
import pytest

from src.saliency import (
    compute_label_aurocs,
    error_cases_to_frame,
    lowest_auroc_labels,
    select_error_cases,
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
