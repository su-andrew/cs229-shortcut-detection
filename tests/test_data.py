import sys

import numpy as np
import pandas as pd

import src.data as data
from src.data import CheXpertValidDataset, parse_args


def test_chexpert_dataset_infers_common_image_column() -> None:
    metadata = pd.DataFrame({"Path": ["patient/image.png"]})

    assert CheXpertValidDataset._infer_image_column(metadata) == "Path"


def test_chexpert_dataset_builds_relative_image_index(tmp_path) -> None:
    image_root = tmp_path / "PNG_valid"
    nested = image_root / "patient"
    nested.mkdir(parents=True)
    image_path = nested / "study.png"
    image_path.write_bytes(b"not a real image")

    image_index = CheXpertValidDataset._build_image_index(image_root)

    assert image_index["study.png"] == image_path
    assert image_index["patient/study.png"] == image_path


def test_data_fetch_cli_parses_dataset_ref(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m src.data",
            "fetch",
            "--dataset-ref",
            "owner.chexpert",
            "--no-progress",
        ],
    )

    args = parse_args()

    assert args.command == "fetch"
    assert args.dataset_ref == "owner.chexpert"
    assert args.no_progress


def test_public_load_xray_image_delegates_to_shared_loader(monkeypatch) -> None:
    expected = np.zeros((1, 4, 4), dtype=np.float32)

    def fake_load_xray_image(path):
        assert str(path) == "case.png"
        return expected

    monkeypatch.setattr(data, "_load_xray_image", fake_load_xray_image)

    assert data.load_xray_image("case.png") is expected
