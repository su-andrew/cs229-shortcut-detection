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

from src.data import default_xrv_transform, load_xray_image


CHEXPERT_COMPETITION_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)

PATH_COLUMN_CANDIDATES = ("path", "image_path", "Path")
XRV_PATHOLOGY_ALIASES = {
    "Pleural Effusion": "Effusion",
}
ATTENTION_LOCATION_OPTIONS = ("inside_lung", "outside_lung", "mixed", "unclear")
SHORTCUT_CATEGORY_OPTIONS = (
    "metadata_overlay",
    "laterality_marker",
    "support_device",
    "acquisition_artifact",
    "image_border",
    "none",
    "unclear",
)
REVIEW_COLUMNS = (
    "gradcam_path",
    "rendered_by_default",
    "attention_location",
    "shortcut_category",
    "include_in_report",
    "notes",
)


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
        # CheXpert: NaN = "not mentioned", -1 = uncertain. Restrict to clean
        # 0/1 before ranking so uncertain rows can't inflate AUROC > 1.0
        # (mirrors the exclusion contract in baseline.py).
        valid = np.isin(y_true, (0.0, 1.0)) & np.isfinite(y_pred)
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
        # Exclude CheXpert uncertain (-1) rows explicitly, matching the
        # baseline.py / compute_label_aurocs exclusion contract.
        frame = frame[frame["y_true"].isin((0.0, 1.0))]
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


def slugify(value: str) -> str:
    """Create a filesystem-safe identifier for labels and figure names."""
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in slug.split("_") if part)


def gradcam_filename(case: ErrorCase) -> str:
    """Return the deterministic filename used for a case's Grad-CAM PNG."""
    safe_label = slugify(case.label)
    return f"{safe_label}_{case.error_type}_{case.rank}.png"


def review_template_from_cases(
    cases: Iterable[ErrorCase], output_dir: Path, max_figures: int | None = None
) -> pd.DataFrame:
    """Create a manual review sheet for eyeballing rendered Grad-CAM figures."""
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        row = dict(case.__dict__)
        row.update(
            {
                "gradcam_path": str(output_dir / gradcam_filename(case)),
                "rendered_by_default": (
                    max_figures is None or index < max_figures
                ),
                "attention_location": "",
                "shortcut_category": "",
                "include_in_report": "",
                "notes": "",
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _normalise_review_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _truthy_review_value(value: object) -> bool:
    return _normalise_review_value(value) in {"1", "true", "yes", "y"}


def summarize_review_annotations(review: pd.DataFrame) -> dict[str, object]:
    """Summarize a filled manual review CSV into milestone-ready counts."""
    missing = [column for column in REVIEW_COLUMNS if column not in review.columns]
    if missing:
        raise ValueError(
            "Review CSV is missing required columns: " + ", ".join(missing)
        )

    reviewed = review[
        review["attention_location"].map(_normalise_review_value).isin(
            ATTENTION_LOCATION_OPTIONS
        )
    ].copy()
    attention_locations = reviewed["attention_location"].map(_normalise_review_value)
    shortcut_categories = reviewed["shortcut_category"].map(_normalise_review_value)
    valid_shortcut_categories = shortcut_categories[
        shortcut_categories.isin(SHORTCUT_CATEGORY_OPTIONS)
        & ~shortcut_categories.isin({"none", "unclear"})
    ]

    outside_lung = int((attention_locations == "outside_lung").sum())
    outside_or_mixed = int(
        attention_locations.isin({"outside_lung", "mixed"}).sum()
    )

    return {
        "total_selected": int(len(review)),
        "reviewed": int(len(reviewed)),
        "outside_lung": outside_lung,
        "outside_or_mixed": outside_or_mixed,
        "included_for_report": int(
            review["include_in_report"].map(_truthy_review_value).sum()
        ),
        "shortcut_category_counts": valid_shortcut_categories.value_counts().to_dict(),
        "label_counts": reviewed["label"].value_counts().to_dict()
        if "label" in reviewed.columns
        else {},
    }


def format_review_summary(summary: dict[str, object]) -> str:
    """Format review counts as a concise paragraph for the milestone draft."""
    reviewed = int(summary["reviewed"])
    total_selected = int(summary["total_selected"])
    outside_or_mixed = int(summary["outside_or_mixed"])
    included = int(summary["included_for_report"])
    category_counts = summary["shortcut_category_counts"]

    if not reviewed:
        return (
            f"No Grad-CAM cases have been manually reviewed yet "
            f"({total_selected} selected)."
        )

    if isinstance(category_counts, dict) and category_counts:
        category_text = ", ".join(
            f"{category.replace('_', ' ')} ({count})"
            for category, count in category_counts.items()
        )
    else:
        category_text = "no specific shortcut category marked yet"

    return (
        f"Manual Grad-CAM review inspected {reviewed} of {total_selected} selected "
        f"high-confidence errors. {outside_or_mixed} showed peak or mixed attention "
        f"outside the lung field. Marked shortcut cues: {category_text}. "
        f"{included} case(s) were flagged for the milestone figure."
    )


def resolve_image_path(image_root: Path, prediction_path: str) -> Path:
    """Resolve a predictions CSV image path against the local image root.

    baseline.py writes repo-root-relative paths (e.g.
    ``data/chexpert/PNG_valid/patient.../view.png``). When run from the repo
    root that path resolves directly. When ``--image-root`` is given
    explicitly, a naive ``image_root / path`` double-prefixes; fall back to
    joining only the patient-relative tail so an explicit root still works.
    """
    path = Path(prediction_path)
    if path.is_absolute() or path.exists():
        return path

    direct = image_root / path
    if direct.exists():
        return direct

    try:
        tail = path.relative_to(image_root)
    except ValueError:
        tail = None
    if tail is not None:
        rerooted = image_root / tail
        if rerooted.exists():
            return rerooted

    return direct


def model_target_label(label: str, pathologies: Iterable[str]) -> str:
    """Map a CheXpert label to the corresponding TorchXRayVision output name."""
    pathologies = list(pathologies)
    if label in pathologies:
        return label

    alias = XRV_PATHOLOGY_ALIASES.get(label)
    if alias in pathologies:
        return alias

    raise ValueError(f"Could not find target label {label!r} in model pathologies.")


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
    """Load a chest X-ray tensor with the shared data-loader preprocessing."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Image loading needs torch installed. Run the project environment setup."
        ) from exc

    try:
        image = load_xray_image(image_path)
        image = default_xrv_transform()(image)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Image loading needs torchxrayvision, torchvision, scikit-image, "
            "and pydicom installed. Run the project environment setup."
        ) from exc

    tensor = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
    return tensor.to(torch.device(device))


def compute_gradcam(model, image_tensor, target_label, model_pathologies=None):
    """Return the raw (224,224) Grad-CAM grayscale map for a target label.

    image_tensor: batched torch tensor (1,1,224,224), same preprocessing as the
    classifier input. Reuses the existing label-aliasing + norm5 target lines.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Grad-CAM needs pytorch-grad-cam installed."
        ) from exc

    pathologies = model_pathologies or getattr(model, "pathologies", None)
    if pathologies is None:
        raise ValueError("Model does not expose a pathologies list.")

    xrv_label = model_target_label(target_label, pathologies)
    target_index = list(pathologies).index(xrv_label)
    target_layers = [model.features.norm5]
    targets = [ClassifierOutputTarget(target_index)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
    return grayscale_cam


def compute_ig(model, image_tensor, target_label, model_pathologies=None,
               n_steps: int = 32):
    """Return a (224,224) Integrated Gradients attribution map for a label.

    NOT VALIDATED -- DO NOT USE FOR REPORTED RESULTS. This was an attempted
    Grad-CAM cross-check, but two issues make its OLL numbers untrustworthy:
    The dominant bug is (1): it takes |attribution|, conflating for/against-class
    evidence (Grad-CAM is ReLU'd, positive-only). A secondary issue is (2): it
    attributes through the xrv model's full forward, whose op_threshs output
    normalization rescales gradient magnitudes non-uniformly across the operating
    threshold (the map is monotone, so the gradient *sign* is preserved -- this
    is a magnitude, not a sign, distortion); IG on this model is cleaner if
    targeted at raw logits via a RawLogits wrapper. Diagnosed 2026-06-01: as-is,
    IG is spuriously anti-correlated with Grad-CAM (r ~ -0.1 to -0.3), driven
    mainly by the abs(); a raw-logit + positive-only fix flips it positive but
    still only weakly/inconsistently agrees (real method difference at 7x7 vs
    pixel resolution). The OLL IG cross-check is left to future work. Kept here
    only as a starting point for that fix.

    Cross-check for compute_gradcam: same signature and same (224,224) output
    contract, so it drops into attribution_outside_mask / the OLL screen
    unchanged. Uses Captum IntegratedGradients with a zero baseline, takes the
    per-pixel |attribution| summed over channels, and min-max normalizes to
    [0,1] to match Grad-CAM's grayscale range. n_steps trades accuracy for
    compute (IG is ~n_steps forward+backward passes, so this is GPU-favorable).
    """
    try:
        import torch
        from captum.attr import IntegratedGradients
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Integrated Gradients needs captum installed (pip install captum)."
        ) from exc

    pathologies = model_pathologies or getattr(model, "pathologies", None)
    if pathologies is None:
        raise ValueError("Model does not expose a pathologies list.")
    xrv_label = model_target_label(target_label, pathologies)
    target_index = list(pathologies).index(xrv_label)

    # IG needs gradients w.r.t. the input; xrv forward applies op_threshs, but
    # IG attributes the chosen output index either way (monotone post-process
    # does not change which pixels matter for that class score's ranking).
    inp = image_tensor.clone().detach().requires_grad_(True)
    baseline = torch.zeros_like(inp)  # plain (non-grad) zero baseline, same device
    ig = IntegratedGradients(model)
    attributions = ig.attribute(inp, target=target_index,
                                baselines=baseline, n_steps=n_steps)
    # (1,1,224,224) -> (224,224): abs, sum channels, normalize to [0,1]
    attr = attributions.detach().abs().sum(dim=1)[0].cpu().numpy()
    rng = attr.max() - attr.min()
    if rng > 0:
        attr = (attr - attr.min()) / rng
    return attr.astype("float32")


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
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Grad-CAM rendering needs matplotlib and pytorch-grad-cam installed."
        ) from exc

    grayscale_cam = compute_gradcam(
        model, image_tensor, target_label, model_pathologies=model_pathologies
    )

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

        output_path = output_dir / gradcam_filename(case)
        image_tensor = load_xray_tensor(image_path, device=device)
        render_gradcam(model, image_tensor, case.label, output_path)
        written.append(output_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and render Grad-CAM errors.")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
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
    parser.add_argument(
        "--review-csv",
        type=Path,
        help="Filled manual_review_template.csv to summarize after visual review.",
    )
    parser.add_argument(
        "--summarize-review",
        action="store_true",
        help="Summarize a filled review CSV and skip selection/rendering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize_review:
        if args.review_csv is None:
            raise SystemExit("--review-csv is required with --summarize-review")
        review = pd.read_csv(args.review_csv)
        summary_text = format_review_summary(summarize_review_annotations(review))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "review_summary.txt"
        summary_path.write_text(summary_text + "\n", encoding="utf-8")
        print(summary_text)
        print(f"Wrote review summary to {summary_path}")
        return

    if args.predictions_csv is None:
        raise SystemExit(
            "--predictions-csv is required unless --summarize-review is used"
        )

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
    review_path = args.output_dir / "manual_review_template.csv"
    review_template_from_cases(cases, args.output_dir, args.max_figures).to_csv(
        review_path, index=False
    )

    print(f"Auditing labels: {', '.join(audit_labels)}")
    print(f"Wrote selected errors to {selected_path}")
    print(f"Wrote manual review template to {review_path}")

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
