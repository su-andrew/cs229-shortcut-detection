import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (
    _chexpert_path_stem,
    _filter_valid_master,
    _parse_chexbert_jsonl,
    CHEXPERT_COMPETITION_LABELS,
    CHEXPERT_DEMOGRAPHIC_COLUMNS,
    CheXpertValidDataset,
    merge_chexpert_sources,
)


def _label_record(path, **labels):
    rec = {"path_to_image": path}
    for name in CHEXPERT_COMPETITION_LABELS:
        rec[name] = labels.get(name, None)
    rec["Pneumonia"] = labels.get("Pneumonia", None)  # non-competition col
    return rec


def test_parse_chexbert_jsonl_keeps_valid_rows_and_competition_labels():
    lines = [
        json.dumps(_label_record("train/patientA/study1/view1_frontal.jpg", Edema=1.0)),
        json.dumps(_label_record("valid/patient64620/study1/view1_frontal.jpg",
                                  Atelectasis=1.0, Cardiomegaly=0.0,
                                  Consolidation=-1.0)),
        "",  # blank line tolerated
        json.dumps(_label_record("valid/patient64678/study1/view1_frontal.jpg")),
    ]
    df = _parse_chexbert_jsonl(lines)

    assert list(df.columns) == ["path_to_image", *CHEXPERT_COMPETITION_LABELS]
    assert "Pneumonia" not in df.columns
    assert len(df) == 2  # train row dropped
    assert set(df["path_to_image"]) == {
        "valid/patient64620/study1/view1_frontal.jpg",
        "valid/patient64678/study1/view1_frontal.jpg",
    }
    row = df[df.path_to_image.str.contains("64620")].iloc[0]
    assert row["Atelectasis"] == 1.0
    assert row["Consolidation"] == -1.0
    # not-mentioned stays NaN
    assert np.isnan(df[df.path_to_image.str.contains("64678")].iloc[0]["Edema"])


def test_parse_chexbert_jsonl_raises_on_malformed_json():
    with pytest.raises(json.JSONDecodeError):
        _parse_chexbert_jsonl(["not-json"])


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("valid/patient64620/study1/view1_frontal.jpg", "patient64620/study1/view1_frontal"),
        ("patient64643/study1/view1_frontal.png", "patient64643/study1/view1_frontal"),
        ("train/patient42142/study5/view1_frontal.jpg", "train/patient42142/study5/view1_frontal"),
        ("valid/patient1/study1/view2_lateral.png", "patient1/study1/view2_lateral"),
    ],
)
def test_chexpert_path_stem_normalizes_prefix_and_extension(raw, expected):
    assert _chexpert_path_stem(raw) == expected


def test_filter_valid_master_selects_valid_split_and_columns():
    raw = pd.DataFrame(
        {
            "path_to_image": [
                "train/patientX/study1/view1_frontal.jpg",
                "valid/patient64620/study1/view1_frontal.jpg",
            ],
            "split": ["train", "valid"],
            "age": [40, 55],
            "sex": ["M", "F"],
            "race": ["White", "Asian"],
            "ethnicity": ["Non-Hispanic", "Hispanic"],
            "insurance_type": ["Medicare", "Private"],
            "frontal_lateral": ["Frontal", "Frontal"],
            "ap_pa": ["AP", "PA"],
            "report": ["...", "..."],  # extra column dropped
        }
    )
    out = _filter_valid_master(raw)

    assert list(out.columns) == [
        "path_to_image",
        "split",
        *CHEXPERT_DEMOGRAPHIC_COLUMNS,
    ]
    assert len(out) == 1
    assert list(out.index) == [0]
    assert out.iloc[0]["path_to_image"].startswith("valid/")
    assert out.iloc[0]["sex"] == "F"


def _images_df(*relpaths):
    return pd.DataFrame({"file_name": list(relpaths)})


def _master_df(rows):
    cols = ["path_to_image", "split", *CHEXPERT_DEMOGRAPHIC_COLUMNS]
    return pd.DataFrame(rows, columns=cols)


def _labels_df(rows):
    cols = ["path_to_image", *CHEXPERT_COMPETITION_LABELS]
    return pd.DataFrame(rows, columns=cols)


def test_merge_joins_three_sources_on_canonical_stem():
    images = _images_df("patient64620/study1/view1_frontal.png")
    master = _master_df(
        [["valid/patient64620/study1/view1_frontal.jpg", "valid",
          55, "F", "Asian", "Hispanic", "Private", "Frontal", "PA"]]
    )
    labels = _labels_df(
        [["valid/patient64620/study1/view1_frontal.jpg",
          1.0, 0.0, -1.0, np.nan, 1.0]]
    )

    out = merge_chexpert_sources(images, master, labels)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["path"] == "patient64620/study1/view1_frontal.png"
    assert row["split"] == "valid"
    assert row["sex"] == "F"
    assert row["Atelectasis"] == 1.0
    assert row["Consolidation"] == -1.0
    assert np.isnan(row["Edema"])
    assert list(out.columns) == [
        "path", "split", *CHEXPERT_COMPETITION_LABELS,
        *CHEXPERT_DEMOGRAPHIC_COLUMNS,
    ]


def test_merge_keeps_image_with_missing_label_as_nan():
    images = _images_df(
        "patient64620/study1/view1_frontal.png",
        "patient64678/study1/view1_frontal.png",
    )
    master = _master_df(
        [["valid/patient64620/study1/view1_frontal.jpg", "valid",
          55, "F", "Asian", "Hispanic", "Private", "Frontal", "PA"]]
    )
    labels = _labels_df(
        [["valid/patient64620/study1/view1_frontal.jpg",
          1.0, 0.0, 0.0, 0.0, 1.0]]
    )

    out = merge_chexpert_sources(images, master, labels)

    assert len(out) == 2  # both images retained
    missing = out[out.path.str.contains("64678")].iloc[0]
    assert pd.isna(missing["Cardiomegaly"])
    assert pd.isna(missing["sex"])


def test_merge_excludes_labels_without_an_image_on_disk():
    images = _images_df("patient64620/study1/view1_frontal.png")
    master = _master_df([])
    labels = _labels_df(
        [
            ["valid/patient64620/study1/view1_frontal.jpg", 1, 0, 0, 0, 0],
            ["valid/patient99999/study1/view1_frontal.jpg", 1, 1, 1, 1, 1],
        ]
    )

    out = merge_chexpert_sources(images, master, labels)

    assert len(out) == 1
    assert out.iloc[0]["path"] == "patient64620/study1/view1_frontal.png"


def test_merge_raises_when_no_image_matches_any_label():
    images = _images_df("patient00001/study1/view1_frontal.png")
    master = _master_df([])
    labels = _labels_df(
        [["valid/patient64620/study1/view1_frontal.jpg", 1, 0, 0, 0, 0]]
    )

    with pytest.raises(ValueError, match="0 .* matched"):
        merge_chexpert_sources(images, master, labels)


def test_merge_deduplicates_labels_with_same_stem():
    images = _images_df("patient64620/study1/view1_frontal.png")
    master = _master_df([])
    labels = _labels_df(
        [
            ["valid/patient64620/study1/view1_frontal.jpg", 1, 0, 0, 0, 0],
            ["valid/patient64620/study1/view1_frontal.jpg", 0, 1, 1, 1, 1],
        ]
    )

    out = merge_chexpert_sources(images, master, labels)

    assert len(out) == 1  # one row per image despite duplicate label rows


def test_merge_raises_when_images_empty():
    images = _images_df()
    master = _master_df([])
    labels = _labels_df(
        [["valid/patient64620/study1/view1_frontal.jpg", 1, 0, 0, 0, 0]]
    )

    with pytest.raises(ValueError, match="images DataFrame is empty"):
        merge_chexpert_sources(images, master, labels)


@pytest.mark.skipif(
    not os.getenv("REDIVIS_API_TOKEN"),
    reason="needs REDIVIS_API_TOKEN + CheXpert Plus Redivis access",
)
def test_fetch_chexpert_valid_smoke(tmp_path):
    from src.data import fetch_chexpert_valid

    paths = fetch_chexpert_valid(
        output_dir=tmp_path,
        dataset_ref="AIMI.chexpert_plus",
        progress=False,
    )
    meta = pd.read_csv(paths.metadata_csv)
    assert {"path", "split", "Atelectasis", "race"}.issubset(meta.columns)
    assert len(meta) > 100  # ~234 valid images
    assert Path(paths.image_dir).exists()


def test_dataset_docstring_does_not_claim_radiologist_gold():
    doc = CheXpertValidDataset.__doc__ or ""
    assert "radiologist-labeled validation set is 0/1 only" not in doc
    assert "CheXbert" in doc  # documents the real label provenance
