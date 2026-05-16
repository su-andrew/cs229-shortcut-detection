"""Data loading utilities for chest X-ray shortcut experiments."""

from __future__ import annotations

import argparse
import argparse
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


CHEXPERT_DATA_DIR = Path("data/chexpert")
CHEXPERT_IMAGE_DIRNAME = "PNG_valid"
CHEXPERT_METADATA_FILENAME = "metadata.csv"

# The five CheXpert competition labels. Default label set for the dataset so
# the `sample["labels"]` contract holds without the caller having to pass them.
CHEXPERT_COMPETITION_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

CHEXPERT_MASTER_TABLE = "df_chexpert_plus_240401"
CHEXPERT_LABELS_TABLE = "CheXpert Labels"
# CheXbert label derivation. report_fixed.json's valid-split labels are
# near-random (an upstream CheXpert Plus artifact defect): scoring a
# pretrained xrv DenseNet against it gives chance AUROC, while the exact
# same predictions score ~0.85-0.88 against impression_fixed.json and
# ~0.81-0.94 against findings_fixed.json. Default to impression (best
# AUROC + coverage; PE n=116). Override per task via --label-file.
CHEXPERT_LABEL_FILE = "impression_fixed.json"
CHEXPERT_DEMOGRAPHIC_COLUMNS = [
    "age",
    "sex",
    "race",
    "ethnicity",
    "insurance_type",
    "frontal_lateral",
    "ap_pa",
]


@dataclass(frozen=True)
class CheXpertValidPaths:
    root: Path
    image_dir: Path
    metadata_csv: Path

    def dataset(self, **kwargs: Any) -> CheXpertValidDataset:
        return CheXpertValidDataset(
            metadata_csv=self.metadata_csv,
            image_root=self.image_dir,
            **kwargs,
        )


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load a CheXpert or MIMIC-CXR metadata CSV."""
    return pd.read_csv(csv_path)


def _download_valid_images(
    redivis: Any,
    dataset_ref: str,
    image_table: str,
    image_dir: Path,
    *,
    overwrite: bool,
    progress: bool,
) -> pd.DataFrame:
    table = _redivis_table(redivis, dataset_ref, image_table)
    table.to_directory().download(
        str(image_dir), overwrite=overwrite, progress=progress
    )
    relpaths = [
        str(p.relative_to(image_dir))
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    return pd.DataFrame({"file_name": relpaths})


def _load_master_demographics(
    redivis: Any, dataset_ref: str, master_table: str
) -> pd.DataFrame:
    table = _redivis_table(redivis, dataset_ref, master_table)
    columns = ["path_to_image", "split", *CHEXPERT_DEMOGRAPHIC_COLUMNS]
    frame = table.to_pandas_dataframe(variables=columns, progress=False)
    return _filter_valid_master(frame)


def _load_chexbert_labels(
    redivis: Any,
    dataset_ref: str,
    labels_table: str,
    label_file: str,
) -> pd.DataFrame:
    table = _redivis_table(redivis, dataset_ref, labels_table)
    files = list(table.list_files())
    match = next((f for f in files if f.name == label_file), None)
    if match is None:
        available = ", ".join(sorted(f.name for f in files))
        raise FileNotFoundError(
            f"Label file {label_file!r} not found in table "
            f"{labels_table!r}. Available: {available}"
        )
    # Redivis' file API exposes no incremental reader, so `read(as_text=True)`
    # is an unavoidable single in-memory read of the ~84MB label file. Iterate
    # it via StringIO rather than `.splitlines()` so we don't also materialize
    # a second full list of ~223k line strings.
    text = match.read(as_text=True)
    return _parse_chexbert_jsonl(io.StringIO(text))


def fetch_chexpert_valid(
    output_dir: str | Path = CHEXPERT_DATA_DIR,
    *,
    dataset_ref: str | None = None,
    image_table: str = CHEXPERT_IMAGE_DIRNAME,
    master_table: str = CHEXPERT_MASTER_TABLE,
    labels_table: str = CHEXPERT_LABELS_TABLE,
    label_file: str = CHEXPERT_LABEL_FILE,
    image_dirname: str = CHEXPERT_IMAGE_DIRNAME,
    metadata_filename: str = CHEXPERT_METADATA_FILENAME,
    overwrite: bool = False,
    progress: bool = True,
) -> CheXpertValidPaths:
    """Download the CheXpert Plus validation split from Redivis and write a
    merged ``metadata.csv``.

    CheXpert Plus stores images, demographics, and CheXbert-derived labels
    in three separate Redivis sources; this joins them on a canonical
    ``patient/study/view`` path stem. Labels are CheXbert machine
    extractions from report text (``impression_fixed.json`` by default),
    NOT radiologist gold — expect ``-1`` (uncertain) and ``NaN`` (not
    mentioned) values.
    """
    import redivis

    dataset_ref = dataset_ref or os.getenv("CHEXPERT_REDIVIS_DATASET")
    if dataset_ref is None or len(dataset_ref.split(".")) != 2:
        raise ValueError(
            "A 2-part 'owner.dataset' Redivis reference is required: set "
            "the CHEXPERT_REDIVIS_DATASET env var or pass dataset_ref= "
            f"(e.g. 'AIMI.chexpert_plus'). Got {dataset_ref!r}."
        )

    output_path = Path(output_dir)
    image_dir = output_path / image_dirname
    metadata_csv = output_path / metadata_filename
    output_path.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    images_df = _download_valid_images(
        redivis, dataset_ref, image_table, image_dir,
        overwrite=overwrite, progress=progress,
    )
    master_df = _load_master_demographics(
        redivis, dataset_ref, master_table
    )
    labels_df = _load_chexbert_labels(
        redivis, dataset_ref, labels_table, label_file
    )

    merged = merge_chexpert_sources(images_df, master_df, labels_df)

    # Accurate join count: an image "has a label row" iff its stem is present
    # in the label source — independent of whether the 5 labels are all NaN
    # ("not mentioned"). The old `notna().any()` conflated those.
    img_stems = images_df["file_name"].map(_chexpert_path_stem)
    label_stems = set(labels_df["path_to_image"].map(_chexpert_path_stem))
    n_joined = int(img_stems.isin(label_stems).sum())
    print(f"{n_joined}/{len(images_df)} images joined to a CheXbert label row")

    # Per-label class balance, so an undefined label (e.g. Atelectasis has
    # n_neg=0 under impression_fixed) is visible here, not silently at eval.
    print("per-label class balance (1=pos, 0=neg; -1/NaN = uncertain/not mentioned):")
    for label in CHEXPERT_COMPETITION_LABELS:
        col = merged[label]
        n_pos = int((col == 1.0).sum())
        n_neg = int((col == 0.0).sum())
        n_other = len(merged) - n_pos - n_neg
        warn = "  [WARNING: <2 classes — AUROC undefined]" if (
            n_pos == 0 or n_neg == 0
        ) else ""
        print(
            f"  {label}: pos={n_pos} neg={n_neg} "
            f"uncertain/unlabeled={n_other}{warn}"
        )

    merged.to_csv(metadata_csv, index=False)

    return CheXpertValidPaths(
        root=output_path, image_dir=image_dir, metadata_csv=metadata_csv
    )


class CheXpertValidDataset:
    """Map-style dataset over the fetched CheXpert Plus validation split.

    ``label_columns`` defaults to the five CheXpert competition labels so
    ``sample["labels"]`` is always populated. Labels are CheXbert machine
    extractions from report text (CheXpert Plus does not ship the original
    radiologist-adjudicated valid labels): values include 1/0, -1
    (uncertain), and NaN (not mentioned), in the validation split too.
    Callers computing AUROC must filter non-{0,1} entries per label
    (``baseline.py`` does this).
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
        self.label_columns = (
            list(label_columns)
            if label_columns is not None
            else list(CHEXPERT_COMPETITION_LABELS)
        )
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
            "image_path": str(image_path),
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


def _chexpert_path_stem(path: str) -> str:
    """Canonical join key: drop a leading 'valid/' split prefix and the
    file extension so PNG image names, the master table, and the CheXbert
    label JSONL all collapse to 'patient<ID>/study<N>/view<N>_<view>'."""
    text = str(path)
    if text.startswith("valid/"):
        text = text[len("valid/"):]
    root, _, ext = text.rpartition(".")
    return root if root else text


def _dedup_by_stem(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Collapse rows sharing a ``_stem``. Identical duplicates are dropped;
    duplicates whose other columns disagree raise — silently keeping the
    first would corrupt the join (one image paired with arbitrary metadata
    or, for master, an inflated number of output rows per image)."""
    dups = frame[frame.duplicated(subset=["_stem"], keep=False)]
    if not dups.empty:
        conflicting = sorted(
            stem
            for stem, group in dups.groupby("_stem")
            if len(group.drop_duplicates()) > 1
        )
        if conflicting:
            shown = ", ".join(conflicting[:10])
            more = (
                "" if len(conflicting) <= 10
                else f" (+{len(conflicting) - 10} more)"
            )
            raise ValueError(
                f"{source_name} has conflicting rows for the same "
                f"patient/study/view stem: {shown}{more}"
            )
    return frame.drop_duplicates(subset=["_stem"], keep="first")


def _parse_chexbert_jsonl(lines: Iterable[str]) -> pd.DataFrame:
    """Parse CheXbert JSON-Lines into a DataFrame, keeping only valid-split
    rows and the 'path_to_image' + 5 competition-label columns. Iterates
    the given lines once; values stay raw (1.0 / 0.0 / -1.0 / NaN)."""
    keep = ["path_to_image", *CHEXPERT_COMPETITION_LABELS]
    records = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        if not str(obj.get("path_to_image", "")).startswith("valid/"):
            continue
        records.append({col: obj.get(col) for col in keep})

    frame = pd.DataFrame(records, columns=keep)
    for col in CHEXPERT_COMPETITION_LABELS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _filter_valid_master(master: pd.DataFrame) -> pd.DataFrame:
    """Keep valid-split rows and the path + demographic columns from the
    CheXpert Plus master metadata table."""
    columns = ["path_to_image", "split", *CHEXPERT_DEMOGRAPHIC_COLUMNS]
    return master.loc[master["split"] == "valid", columns].reset_index(drop=True)


def merge_chexpert_sources(
    images: pd.DataFrame,
    master: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join the three CheXpert Plus sources on a canonical path stem.

    Images on disk are authoritative: one output row per image file.
    Missing master/label data is left as NaN. Raises if no image matches
    any label (signals a path-normalization regression).
    """
    img = images.copy()
    img["_stem"] = img["file_name"].map(_chexpert_path_stem)
    img["path"] = img["file_name"].astype(str)

    mas = master.copy()
    mas["_stem"] = mas["path_to_image"].map(_chexpert_path_stem)
    mas = mas.drop(columns=["path_to_image"])
    mas = _dedup_by_stem(mas, "master demographics")

    lab = labels.copy()
    lab["_stem"] = lab["path_to_image"].map(_chexpert_path_stem)
    lab = lab.drop(columns=["path_to_image"])
    lab = _dedup_by_stem(lab, "CheXbert labels")

    if len(img) == 0:
        raise ValueError(
            "images DataFrame is empty — no image files were found "
            "(check the image download)."
        )

    matched = img["_stem"].isin(set(lab["_stem"]))
    if not matched.any():
        raise ValueError(
            f"0 of {len(img)} images matched any label row — path "
            "normalization is likely broken (check _chexpert_path_stem)."
        )

    merged = (
        img.merge(lab, on="_stem", how="left")
        .merge(mas, on="_stem", how="left")
    )

    ordered = [
        "path",
        "split",
        *CHEXPERT_COMPETITION_LABELS,
        *CHEXPERT_DEMOGRAPHIC_COLUMNS,
    ]
    return merged[ordered].reset_index(drop=True)


def _default_xrv_transform() -> Callable[[Any], Any]:
    import torchxrayvision as xrv
    import torchvision

    return torchvision.transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ]
    )


def default_xrv_transform() -> Callable[[Any], Any]:
    """Return the standard TorchXRayVision crop/resize preprocessing."""
    return _default_xrv_transform()


def load_xray_image(image_path: str | Path) -> Any:
    """Load and normalize a raster or DICOM chest X-ray as ``[1, H, W]``."""
    return _load_xray_image(Path(image_path))


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

    image = imread(image_path)
    if image.ndim == 3:
        # Drop an alpha channel before collapsing to grayscale — otherwise
        # the (typically 255) alpha gets averaged into every pixel. dtype is
        # preserved so the integer branch of _image_max_value still applies.
        if image.shape[2] == 4:
            image = image[..., :3]
        image = image.mean(axis=2).astype(image.dtype)
    return image


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

    # Float images (e.g. DICOM after rescale slope/intercept) can span an
    # arbitrary range; the bucketed heuristic crushed HU images to near-black.
    # Use the actual maximum.
    return float(np.nanmax(image))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.data",
        description="CheXpert data utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download the CheXpert Plus validation split (images + labels) from Redivis.",
        description=(
            "Pull the CheXpert Plus validation images, demographics, and CheXbert "
            "labels from Redivis into a local directory. Requires REDIVIS_API_TOKEN "
            "to be set in the environment. The dataset reference must be the 2-part "
            "'owner.dataset' form (e.g. AIMI.chexpert_plus)."
        ),
    )
    fetch_parser.add_argument(
        "--dataset-ref",
        default=None,
        help=(
            "2-part 'owner.dataset' Redivis reference. Falls back to the "
            "CHEXPERT_REDIVIS_DATASET env var."
        ),
    )
    fetch_parser.add_argument(
        "--image-table",
        default=CHEXPERT_IMAGE_DIRNAME,
        help=f"Redivis table name for the validation images (default: {CHEXPERT_IMAGE_DIRNAME}).",
    )
    fetch_parser.add_argument(
        "--master-table",
        default=CHEXPERT_MASTER_TABLE,
        help=(
            f"Redivis table name for the master demographics table "
            f"(default: {CHEXPERT_MASTER_TABLE})."
        ),
    )
    fetch_parser.add_argument(
        "--labels-table",
        default=CHEXPERT_LABELS_TABLE,
        help=(
            f"Redivis table name for the CheXbert labels table "
            f"(default: {CHEXPERT_LABELS_TABLE})."
        ),
    )
    fetch_parser.add_argument(
        "--label-file",
        default=CHEXPERT_LABEL_FILE,
        help=f"Filename of the CheXbert JSONL file within the labels table (default: {CHEXPERT_LABEL_FILE}).",
    )
    fetch_parser.add_argument(
        "--output-dir",
        default=str(CHEXPERT_DATA_DIR),
        help=f"Local output directory (default: {CHEXPERT_DATA_DIR}).",
    )
    fetch_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files instead of skipping them.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if args.command == "fetch":
        paths = fetch_chexpert_valid(
            output_dir=args.output_dir,
            dataset_ref=args.dataset_ref,
            image_table=args.image_table,
            master_table=args.master_table,
            labels_table=args.labels_table,
            label_file=args.label_file,
            overwrite=args.overwrite,
        )
        print("Fetched CheXpert Plus validation data:")
        print(f"  images:   {paths.image_dir}")
        print(f"  metadata: {paths.metadata_csv}")


if __name__ == "__main__":
    main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CheXpert data utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Download CheXpert validation images and metadata from Redivis."
    )
    fetch_parser.add_argument("--output-dir", type=Path, default=CHEXPERT_DATA_DIR)
    fetch_parser.add_argument("--dataset-ref")
    fetch_parser.add_argument("--image-table")
    fetch_parser.add_argument("--metadata-table")
    fetch_parser.add_argument("--image-dirname", default=CHEXPERT_IMAGE_DIRNAME)
    fetch_parser.add_argument("--metadata-filename", default=CHEXPERT_METADATA_FILENAME)
    fetch_parser.add_argument("--file-id-variable")
    fetch_parser.add_argument("--file-name-variable")
    fetch_parser.add_argument("--overwrite", action="store_true")
    fetch_parser.add_argument("--no-progress", action="store_true")

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate that a fetched CheXpert validation set is loadable."
    )
    inspect_parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=CHEXPERT_DATA_DIR / CHEXPERT_METADATA_FILENAME,
    )
    inspect_parser.add_argument(
        "--image-root",
        type=Path,
        default=CHEXPERT_DATA_DIR / CHEXPERT_IMAGE_DIRNAME,
    )
    inspect_parser.add_argument("--image-column")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "fetch":
        paths = fetch_chexpert_valid(
            output_dir=args.output_dir,
            dataset_ref=args.dataset_ref,
            image_table=args.image_table,
            metadata_table=args.metadata_table,
            image_dirname=args.image_dirname,
            metadata_filename=args.metadata_filename,
            file_id_variable=args.file_id_variable,
            file_name_variable=args.file_name_variable,
            overwrite=args.overwrite,
            progress=not args.no_progress,
        )
        print(f"Wrote metadata to {paths.metadata_csv}")
        print(f"Wrote images under {paths.image_dir}")
        return

    if args.command == "inspect":
        dataset = CheXpertValidDataset(
            metadata_csv=args.metadata_csv,
            image_root=args.image_root,
            image_column=args.image_column,
            transform=None,
        )
        print(f"Loaded {len(dataset)} metadata row(s)")
        if len(dataset):
            sample = dataset[0]
            print(f"First image: {sample['image_path']}")
            print(f"Image tensor shape: {tuple(sample['image'].shape)}")
            print(f"Label shape: {tuple(sample['labels'].shape)}")
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
