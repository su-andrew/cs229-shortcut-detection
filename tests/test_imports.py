from src.data import load_metadata
from src.interpretability import attribution_outside_mask
from src.saliency import select_error_cases
from src.shortcuts import SHORTCUT_CATEGORIES, shortcut_reliance_score
from src.utils import set_seed


def test_imports() -> None:
    assert load_metadata is not None
    assert attribution_outside_mask is not None
    assert select_error_cases is not None
    assert "metadata_overlay" in SHORTCUT_CATEGORIES
    assert shortcut_reliance_score is not None
    assert set_seed is not None
