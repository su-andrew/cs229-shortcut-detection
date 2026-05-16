"""Baseline (A1): pretrained DenseNet-121 inference -> per-disease AUROC.

No training. Runs the CheXpert-pretrained TorchXRayVision DenseNet-121 over
the CheXpert validation split, computes per-label AUROC on the five CheXpert
competition labels, and writes two artifacts:

  results/auroc.csv            label, auroc, n_pos, n_neg, n_excluded,
                               published_ref_approx
  results/predictions_val.csv  path, {label}_true, {label}_pred

The predictions CSV schema (`path`, ``{label}_true``, ``{label}_pred`` for
each of the five competition labels) is the Step 4 contract consumed by
``src/saliency.py``.

Run:
  python -m src.data fetch                       # -> data/chexpert/
  python -m src.baseline --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import CheXpertValidDataset
from src.models import XRV_PATHOLOGY_ALIASES, build_model
from src.utils import set_seed

# Approximate published CheXpert competition AUROCs (Irvin et al. 2019),
# used ONLY as a sanity anchor to catch a broken pipeline (label
# misalignment, bad normalization). These are NOT our results and must not
# be cited as such — the pretrained xrv model will not match these exactly.
PUBLISHED_CHEXPERT_AUROC_APPROX = {
    "Atelectasis": 0.85,
    "Cardiomegaly": 0.83,
    "Consolidation": 0.90,
    "Edema": 0.93,
    "Pleural Effusion": 0.93,
}

# Warn (don't fail) if our AUROC is this far from the published anchor —
# a large gap almost always means label misalignment or wrong normalization.
SANITY_TOLERANCE = 0.15


def load_config(path: str | Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def select_device(requested: str):
    """Resolve a torch device. 'auto' prefers cuda, then mps, then cpu."""
    import torch

    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _map_labels_to_indices(labels: list[str], pathologies: list[str]) -> dict[str, int]:
    """Map each competition label to its index in the model's output vector.

    Handles the ``Pleural Effusion`` -> ``Effusion`` xrv naming mismatch.
    """
    label_to_index: dict[str, int] = {}
    for label in labels:
        xrv_name = XRV_PATHOLOGY_ALIASES.get(label, label)
        if xrv_name not in pathologies:
            raise KeyError(
                f"Label {label!r} (xrv name {xrv_name!r}) is not in "
                f"model.pathologies: {pathologies}"
            )
        label_to_index[label] = pathologies.index(xrv_name)
    return label_to_index


def run_baseline(
    config: dict,
    data_root: Path | None = None,
    device_str: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run inference + scoring. Returns (auroc_df, predictions_df)."""
    import torch
    from sklearn.metrics import roc_auc_score

    labels = list(config["data"]["label_columns"])
    set_seed(int(config.get("seed", 229)))

    model = build_model(config["model"]["name"])
    device = select_device(device_str)
    model = model.to(device)

    label_to_index = _map_labels_to_indices(labels, list(model.pathologies))

    # Pass label_columns explicitly so the dataset emits sample["labels"]
    # (it defaults to None, per Codex review of PR #3).
    if data_root is not None:
        root = Path(data_root)
        dataset = CheXpertValidDataset(
            metadata_csv=root / "metadata.csv",
            image_root=root / "PNG_valid",
            label_columns=labels,
        )
    else:
        dataset = CheXpertValidDataset(label_columns=labels)

    n = len(dataset)
    if n == 0:
        raise RuntimeError(
            "CheXpert validation set is empty. Run `python -m src.data fetch` "
            "to populate data/chexpert/ first."
        )

    rows: list[dict] = []
    print(f"Running baseline over {n} validation images on {device} ...")
    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            image = sample["image"]
            if not torch.is_tensor(image):
                image = torch.as_tensor(np.asarray(image), dtype=torch.float32)
            # dataset yields [1, 224, 224]; model wants [B, 1, 224, 224]
            image = image.float().unsqueeze(0).to(device)
            probs = model(image)[0].detach().cpu().numpy()  # xrv: sigmoid outputs

            true = np.asarray(sample["labels"], dtype="float32")
            row = {"path": str(sample["image_path"])}
            for j, label in enumerate(labels):
                row[f"{label}_true"] = float(true[j])
                row[f"{label}_pred"] = float(probs[label_to_index[label]])
            rows.append(row)

            if (i + 1) % 25 == 0 or (i + 1) == n:
                print(f"  {i + 1}/{n}")

    predictions = pd.DataFrame(rows)

    auroc_rows = []
    for label in labels:
        y_true = predictions[f"{label}_true"].to_numpy()
        y_pred = predictions[f"{label}_pred"].to_numpy()
        # CheXpert: NaN = "not mentioned", -1 = uncertain. The radiologist-
        # labeled val set is 0/1 only, but exclude anything that isn't a
        # clean 0/1 so a stray uncertain/NaN can't crash roc_auc_score.
        valid = np.isin(y_true, (0.0, 1.0)) & np.isfinite(y_pred)
        y_true_v = y_true[valid].astype(int)
        y_pred_v = y_pred[valid].astype(float)
        n_pos = int((y_true_v == 1).sum())
        n_neg = int((y_true_v == 0).sum())
        n_excluded = int((~valid).sum())

        if n_pos == 0 or n_neg == 0:
            auroc = float("nan")  # AUROC undefined with a single class
        else:
            auroc = float(roc_auc_score(y_true_v, y_pred_v))

        ref = PUBLISHED_CHEXPERT_AUROC_APPROX.get(label, float("nan"))
        auroc_rows.append(
            {
                "label": label,
                "auroc": auroc,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_excluded": n_excluded,
                "published_ref_approx": ref,
            }
        )
        if (
            np.isfinite(auroc)
            and np.isfinite(ref)
            and abs(auroc - ref) > SANITY_TOLERANCE
        ):
            print(
                f"  [sanity] {label}: AUROC {auroc:.3f} vs published "
                f"~{ref:.2f} (|delta| > {SANITY_TOLERANCE}). Check label "
                f"alignment / normalization before trusting this number."
            )

    auroc_df = pd.DataFrame(auroc_rows)
    return auroc_df, predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CheXpert pretrained-DenseNet baseline -> per-disease AUROC (A1)."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override CheXpert data dir (expects metadata.csv + PNG_valid/).",
    )
    parser.add_argument(
        "--device", default="auto", help="auto | cpu | mps | cuda"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    auroc_df, predictions = run_baseline(
        config, data_root=args.data_root, device_str=args.device
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    auroc_path = args.results_dir / "auroc.csv"
    pred_path = args.results_dir / "predictions_val.csv"
    auroc_df.to_csv(auroc_path, index=False)
    predictions.to_csv(pred_path, index=False)

    print("\nPer-disease AUROC:")
    print(auroc_df.to_string(index=False))
    print(f"\nWrote {auroc_path}")
    print(f"Wrote {pred_path}  ({len(predictions)} rows)")


if __name__ == "__main__":
    main()
