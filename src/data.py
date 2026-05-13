"""Data loading placeholders for chest X-ray shortcut experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load a CheXpert or MIMIC-CXR metadata CSV."""
    return pd.read_csv(csv_path)
