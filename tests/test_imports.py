from src.data import (
    CheXpertValidDataset,
    fetch_chexpert_valid,
    load_metadata,
    merge_chexpert_sources,
)
from src.interpretability import attribution_outside_mask
from src.shortcuts import SHORTCUT_CATEGORIES, shortcut_reliance_score
from src.utils import set_seed


def test_imports() -> None:
    assert load_metadata is not None
    assert fetch_chexpert_valid is not None
    assert CheXpertValidDataset is not None
    assert attribution_outside_mask is not None
    assert "metadata_overlay" in SHORTCUT_CATEGORIES
    assert shortcut_reliance_score is not None
    assert set_seed is not None
    assert merge_chexpert_sources is not None
