"""Data loading utilities for chest X-ray shortcut experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

try:
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - keeps metadata utilities importable
    TorchDataset = object


CHEXPERT_DATA_DIR = Path("data/chexpert")
CHEXPERT_IMAGE_DIRNAME = "PNG_valid"
CHEXPERT_METADATA_FILENAME = "metadata.csv"


@dataclass(frozen=True)
class CheXpertValidPaths:
    """Local files produced by :func:`fetch_chexpert_valid`."""

    root: Path
    image_dir: Path
    metadata_csv: Path

    def dataset(self, **kwargs: Any) -> CheXpertValidDataset:
        """Construct a one-line loadable validation dataset from fetched files."""
        return CheXpertValidDataset(
            metadata_csv=self.metadata_csv,
            image_root=self.image_dir,
            **kwargs,
        )


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load a CheXpert or MIMIC-CXR metadata CSV."""
    return pd.read_csv(csv_path)


def fetch_chexpert_valid(
    output_dir: str | Path = CHEXPERT_DATA_DIR,
    *,
    dataset_ref: str | None = None,
    image_table: str | None = None,
    metadata_table: str | None = None,
    image_dirname: str = CHEXPERT_IMAGE_DIRNAME,
    metadata_filename: str = CHEXPERT_METADATA_FILENAME,
    file_id_variable: str | None = None,
    file_name_variable: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> CheXpertValidPaths:
    """Download the CheXpert validation images and metadata from Redivis.

    The exact CheXpert Plus Redivis dataset reference is access-controlled, so
    callers can provide it directly or via ``CHEXPERT_REDIVIS_DATASET``. Table
    names default to ``PNG_valid`` for images and ``metadata`` for the CSV, and
    can be overridden with ``CHEXPERT_REDIVIS_IMAGE_TABLE`` and
    ``CHEXPERT_REDIVIS_METADATA_TABLE``.
    """
    try:
        import redivis
    except ImportError as exc:  # pragma: no cover - depends on local env setup
        raise ImportError(
            "Install the `redivis` package or update the project environment "
            "before fetching CheXpert data."
        ) from exc

    dataset_ref = dataset_ref or os.getenv("CHEXPERT_REDIVIS_DATASET")
    image_table = image_table or os.getenv(
        "CHEXPERT_REDIVIS_IMAGE_TABLE", CHEXPERT_IMAGE_DIRNAME
    )
    metadata_table = metadata_table or os.getenv(
        "CHEXPERT_REDIVIS_METADATA_TABLE", "metadata"
    )

    output_path = Path(output_dir)
    image_dir = output_path / image_dirname
    metadata_csv = output_path / metadata_filename
    output_path.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_table_ref = _redivis_table(redivis, dataset_ref, image_table)
    metadata_table_ref = _redivis_table(redivis, dataset_ref, metadata_table)

    directory_kwargs = _drop_none(
        {
            "file_id_variable": file_id_variable,
            "file_name_variable": file_name_variable,
        }
    )
    image_table_ref.to_directory(**directory_kwargs).download(
        str(image_dir),
        overwrite=overwrite,
    )
    metadata_table_ref.download(
        str(metadata_csv),
        format="csv",
        overwrite=overwrite,
        progress=progress,
    )

    return CheXpertValidPaths(
        root=output_path,
        image_dir=image_dir,
        metadata_csv=metadata_csv,
    )


class CheXpertValidDataset(TorchDataset):
    """Torch-compatible Dataset for the fetched CheXpert validation split.

    Images are loaded from PNG/JPG or DICOM files, converted to one channel, and
    normalized with ``torchxrayvision.datasets.normalize``.
    """

    DEFAULT_IMAGE_COLUMNS = (
        "path",
        "Path",
        "image_path",
        "ImagePath",
        "file_path",
        "FilePath",
        "file_name",
        "filename",
        "FileName",
    )

    def __init__(
        self,
        metadata_csv: str | Path = CHEXPERT_DATA_DIR / CHEXPERT_METADATA_FILENAME,
        image_root: str | Path = CHEXPERT_DATA_DIR / CHEXPERT_IMAGE_DIRNAME,
        *,
        image_column: str | None = None,
        label_columns: Iterable[str] | None = None,
        transform: Callable[[Any], Any] | None = None,
        include_metadata: bool = False,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.image_root = Path(image_root)
        self.metadata = load_metadata(self.metadata_csv)
        self.image_column = image_column or self._infer_image_column(self.metadata)
        self.label_columns = list(label_columns) if label_columns is not None else None
        self.transform = transform or _default_xrv_transform()
        self.include_metadata = include_metadata
        self._image_index = self._build_image_index(self.image_root)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        image_path = self._resolve_image_path(row[self.image_column])
        image = _load_xray_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        sample: dict[str, Any] = {
            "image": _as_float_tensor(image),
            "image_path": image_path,
        }
        if self.label_columns is not None:
            sample["labels"] = row[self.label_columns].astype("float32").to_numpy()
        if self.include_metadata:
            sample["metadata"] = row.to_dict()

        return sample

    @classmethod
    def _infer_image_column(cls, metadata: pd.DataFrame) -> str:
        for column in cls.DEFAULT_IMAGE_COLUMNS:
            if column in metadata.columns:
                return column

        raise ValueError(
            "Could not infer an image path column. Pass `image_column=` with one "
            f"of the metadata columns: {list(metadata.columns)}"
        )

    @staticmethod
    def _build_image_index(image_root: Path) -> dict[str, Path]:
        if not image_root.exists():
            raise FileNotFoundError(
                f"Image root does not exist: {image_root}. Run "
                "`fetch_chexpert_valid()` first."
            )

        image_index: dict[str, Path] = {}
        for path in image_root.rglob("*"):
            if path.is_file():
                image_index[path.name] = path
                image_index[str(path.relative_to(image_root))] = path

        return image_index

    def _resolve_image_path(self, metadata_value: Any) -> Path:
        candidate = Path(str(metadata_value))

        if candidate.is_absolute() and candidate.exists():
            return candidate

        rooted_candidate = self.image_root / candidate
        if rooted_candidate.exists():
            return rooted_candidate

        indexed_path = self._image_index.get(str(candidate)) or self._image_index.get(
            candidate.name
        )
        if indexed_path is not None:
            return indexed_path

        raise FileNotFoundError(
            f"Could not resolve image path `{metadata_value}` under {self.image_root}."
        )


def _redivis_table(redivis: Any, dataset_ref: str | None, table_ref: str) -> Any:
    if dataset_ref is None:
        return redivis.table(table_ref)

    return redivis.table(f"{dataset_ref}.{table_ref}")


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _default_xrv_transform() -> Callable[[Any], Any]:
    try:
        import torchxrayvision as xrv
        import torchvision
    except ImportError as exc:  # pragma: no cover - depends on local env setup
        raise ImportError(
            "Install `torchxrayvision` and `torchvision` before constructing the "
            "CheXpert dataset."
        ) from exc

    return torchvision.transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ]
    )


def _load_xray_image(image_path: Path) -> Any:
    import torchxrayvision as xrv

    suffix = image_path.suffix.lower()
    if suffix == ".dcm":
        image = _read_dicom_image(image_path)
    else:
        image = _read_raster_image(image_path)

    max_value = _image_max_value(image)
    image = image.astype("float32")

    if image.ndim == 3:
        image = image.mean(axis=2)

    image = xrv.datasets.normalize(image, max_value)
    return image[None, ...]


def _as_float_tensor(image: Any) -> Any:
    import torch

    return torch.as_tensor(image, dtype=torch.float32)


def _read_raster_image(image_path: Path) -> np.ndarray:
    from skimage.io import imread

    return imread(image_path)


def _read_dicom_image(image_path: Path) -> np.ndarray:
    import pydicom

    dicom = pydicom.dcmread(image_path)
    image = dicom.pixel_array.astype("float32")

    slope = float(getattr(dicom, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    return image


def _image_max_value(image: np.ndarray) -> float:
    if np.issubdtype(image.dtype, np.integer):
        return float(np.iinfo(image.dtype).max)

    image_max = float(np.nanmax(image))
    if image_max <= 1.0:
        return 1.0
    if image_max <= 255.0:
        return 255.0
    return 65535.0
