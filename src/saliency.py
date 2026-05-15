"""Grad-CAM saliency audit for milestone error analysis.

This module is intentionally split into two parts:

1. Error-case selection only needs ``results/predictions_val.csv``.
2. Grad-CAM rendering additionally needs image files and the pretrained
   TorchXRayVision model weights.

Expected prediction CSV columns are:
``path`` plus ``{label}_true`` and ``{label}_pred`` for each CheXpert label.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CHEXPERT_COMPETITION_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)

PATH_COLUMN_CANDIDATES = ("path", "image_path", "Path")


@dataclass(frozen=True)
class ErrorCase:
    """One high-confidence classifier error selected for visual audit."""

    label: str
    error_type: str
    path: str
    y_true: int
    y_pred: float
    confidence: float
    rank: int


def true_column(label: str) -> str:
    return f"{label}_true"


def pred_column(label: str) -> str:
    return f"{label}_pred"


def find_path_column(columns: Iterable[str]) -> str:
    """Return the image-path column name used by a predictions CSV."""
    column_set = set(columns)
    for candidate in PATH_COLUMN_CANDIDATES:
        if candidate in column_set:
            return candidate
    raise ValueError(
        "Predictions CSV must include an image path column. Expected one of "
        f"{PATH_COLUMN_CANDIDATES}."
    )


def validate_prediction_columns(
    predictions: pd.DataFrame, labels: Iterable[str]
) -> str:
    """Validate the Step 3 CSV contract and return the path column name."""
    path_column = find_path_column(predictions.columns)
    missing: list[str] = []
    for label in labels:
        for column in (true_column(label), pred_column(label)):
            if column not in predictions.columns:
                missing.append(column)

    if missing:
        raise ValueError(
            "Predictions CSV is missing required columns: " + ", ".join(missing)
        )

    return path_column


def compute_label_aurocs(
    predictions: pd.DataFrame, labels: Iterable[str] = CHEXPERT_COMPETITION_LABELS
) -> pd.Series:
    """Compute per-label AUROC from the Step 3 predictions CSV."""
    validate_prediction_columns(predictions, labels)

    aurocs: dict[str, float] = {}
    for label in labels:
        y_true = predictions[true_column(label)].to_numpy()
        y_pred = predictions[pred_column(label)].to_numpy()
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid].astype(int)
        y_pred = y_pred[valid].astype(float)

        if len(np.unique(y_true)) < 2:
            aurocs[label] = np.nan
        else:
            aurocs[label] = binary_auroc(y_true, y_pred)

    return pd.Series(aurocs, name="auroc").sort_values(na_position="last")


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary AUROC without requiring scikit-learn at import time."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)

    start = 0
    while start < len(sorted_scores):
        stop = start + 1
        while (
            stop < len(sorted_scores)
            and sorted_scores[stop] == sorted_scores[start]
        ):
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop

    positive_rank_sum = ranks[positives].sum()
    return float(
        (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def lowest_auroc_labels(aurocs: pd.Series, n_labels: int) -> list[str]:
    """Return labels with the lowest finite AUROC values."""
    finite_aurocs = aurocs.dropna().sort_values()
    return finite_aurocs.head(n_labels).index.tolist()


def select_error_cases(
    predictions: pd.DataFrame,
    labels: Iterable[str],
    top_k_per_type: int = 5,
    threshold: float = 0.5,
) -> list[ErrorCase]:
    """Pick high-confidence false positives and false negatives for each label.

    False positives are sorted by highest predicted probability. False negatives
    are sorted by lowest predicted probability, which makes the most confident
    misses appear first.
    """
    path_column = validate_prediction_columns(predictions, labels)
    cases: list[ErrorCase] = []

    for label in labels:
        label_true = true_column(label)
        label_pred = pred_column(label)
        frame = predictions[[path_column, label_true, label_pred]].copy()
        frame = frame.rename(
            columns={path_column: "path", label_true: "y_true", label_pred: "y_pred"}
        )
        frame = frame.dropna(subset=["y_true", "y_pred"])
        frame["y_true"] = frame["y_true"].astype(int)
        frame["y_pred"] = frame["y_pred"].astype(float)

        false_positives = frame[
            (frame["y_true"] == 0) & (frame["y_pred"] >= threshold)
        ].sort_values("y_pred", ascending=False)
        false_negatives = frame[
            (frame["y_true"] == 1) & (frame["y_pred"] < threshold)
        ].sort_values("y_pred", ascending=True)

        for rank, row in enumerate(
            false_positives.head(top_k_per_type).itertuples(), 1
        ):
            cases.append(
                ErrorCase(
                    label=label,
                    error_type="false_positive",
                    path=str(row.path),
                    y_true=0,
                    y_pred=float(row.y_pred),
                    confidence=float(row.y_pred),
                    rank=rank,
                )
            )

        for rank, row in enumerate(
            false_negatives.head(top_k_per_type).itertuples(), 1
        ):
            cases.append(
                ErrorCase(
                    label=label,
                    error_type="false_negative",
                    path=str(row.path),
                    y_true=1,
                    y_pred=float(row.y_pred),
                    confidence=float(1.0 - row.y_pred),
                    rank=rank,
                )
            )

    return cases


def error_cases_to_frame(cases: Iterable[ErrorCase]) -> pd.DataFrame:
    """Convert selected errors to a CSV-friendly table."""
    return pd.DataFrame([case.__dict__ for case in cases])


def resolve_image_path(image_root: Path, prediction_path: str) -> Path:
    """Resolve a predictions CSV image path against the local image root."""
    path = Path(prediction_path)
    if path.is_absolute():
        return path
    return image_root / path


def load_chexpert_model(device: str = "cpu"):
    """Load the CheXpert-pretrained TorchXRayVision DenseNet-121."""
    try:
        import torch
        import torchxrayvision as xrv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Grad-CAM rendering needs torch and torchxrayvision installed. "
            "Run the project environment setup first."
        ) from exc

    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model = model.to(torch.device(device))
    model.eval()
    return model


def load_xray_tensor(image_path: Path, device: str = "cpu"):
    """Load a chest X-ray image and apply TorchXRayVision preprocessing."""
    try:
        import skimage.io
        import torch
        import torchxrayvision as xrv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Image loading needs scikit-image, torch, and torchxrayvision installed."
        ) from exc

    image = skimage.io.imread(image_path)
    if image.ndim == 3:
        image = image.mean(axis=2)

    image = xrv.datasets.normalize(image, 255)
    image = image[None, ...]
    image = xrv.datasets.XRayCenterCrop()(image)
    image = xrv.datasets.XRayResizer(224)(image)

    tensor = torch.from_numpy(image).float().unsqueeze(0)
    return tensor.to(torch.device(device))


def render_gradcam(
    model,
    image_tensor,
    target_label: str,
    output_path: Path,
    model_pathologies: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Render one Grad-CAM overlay for a target CheXpert label."""
    try:
        import matplotlib.pyplot as plt
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Grad-CAM rendering needs matplotlib and pytorch-grad-cam installed."
        ) from exc

    pathologies = model_pathologies or getattr(model, "pathologies", None)
    if pathologies is None or target_label not in pathologies:
        raise ValueError(
            f"Could not find target label {target_label!r} in model pathologies."
        )

    target_index = list(pathologies).index(target_label)
    target_layers = [model.features.norm5]
    targets = [ClassifierOutputTarget(target_index)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    image = image_tensor.detach().cpu().numpy()[0, 0]
    image = image - image.min()
    if image.max() > 0:
        image = image / image.max()
    rgb_image = np.repeat(image[..., None], 3, axis=2)
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, overlay)


def render_error_case_figures(
    cases: Iterable[ErrorCase],
    image_root: Path,
    output_dir: Path,
    device: str = "cpu",
    max_figures: int = 4,
) -> list[Path]:
    """Render Grad-CAM overlays for the selected error cases."""
    model = load_chexpert_model(device=device)
    written: list[Path] = []

    for case in list(cases)[:max_figures]:
        image_path = resolve_image_path(image_root, case.path)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image for selected error case does not exist: {image_path}"
            )

        safe_label = case.label.lower().replace(" ", "_")
        output_path = output_dir / f"{safe_label}_{case.error_type}_{case.rank}.png"
        image_tensor = load_xray_tensor(image_path, device=device)
        render_gradcam(model, image_tensor, case.label, output_path)
        written.append(output_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and render Grad-CAM errors.")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="Step 3 predictions CSV with path, *_true, and *_pred columns.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/chexpert"),
        help="Root directory used to resolve relative image paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory for selected_errors.csv and Grad-CAM PNGs.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(CHEXPERT_COMPETITION_LABELS),
        help="Labels to consider from the predictions CSV.",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Number of lowest-AUROC labels to audit.",
    )
    parser.add_argument(
        "--top-k-per-type",
        type=int,
        default=5,
        help="False positives and false negatives to select per audited label.",
    )
    parser.add_argument(
        "--max-figures",
        type=int,
        default=4,
        help="Maximum Grad-CAM overlays to render.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to define FP/FN errors.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for Grad-CAM rendering, e.g. cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Only write selected_errors.csv; skip Grad-CAM rendering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions_csv)
    aurocs = compute_label_aurocs(predictions, args.labels)
    audit_labels = lowest_auroc_labels(aurocs, args.num_labels)
    cases = select_error_cases(
        predictions,
        audit_labels,
        top_k_per_type=args.top_k_per_type,
        threshold=args.threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aurocs.to_csv(args.output_dir / "computed_aurocs.csv", header=True)
    selected_path = args.output_dir / "selected_errors.csv"
    error_cases_to_frame(cases).to_csv(selected_path, index=False)

    print(f"Auditing labels: {', '.join(audit_labels)}")
    print(f"Wrote selected errors to {selected_path}")

    if args.select_only:
        print("Skipped Grad-CAM rendering because --select-only was passed.")
        return

    written = render_error_case_figures(
        cases,
        image_root=args.image_root,
        output_dir=args.output_dir,
        device=args.device,
        max_figures=args.max_figures,
    )
    print(f"Wrote {len(written)} Grad-CAM figure(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
