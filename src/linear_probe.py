"""Linear probe (CS229 non-deep baseline).

Fits a per-disease logistic regression on FROZEN DenseNet-121 penultimate
features (global-average-pooled 1024-d) using a CheXpert train sample, then
evaluates on the held-out 234-image val split — alongside the CNN's own AUROC.

Thesis tie-in: if a *linear* probe on frozen features ≈ the CNN's AUROC, the
disease signal (and any shortcuts) is linearly decodable from the
representation — i.e. the penultimate features already encode whatever the
model relies on. Probe ≪ CNN would instead say the head does real nonlinear
work. Either is a legitimate, reportable finding.

Labels are CheXbert `impression_fixed` extractions, NOT radiologist gold — the
probe inherits that caveat. Train (PNG_train sample) and val (234) are disjoint
CheXpert patient splits, so there is no train/val leakage by construction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import set_seed

MIN_N = 10  # ranking gate: need >=10 pos AND >=10 neg in val to rank a disease


def global_average_pool(features: np.ndarray) -> np.ndarray:
    """(N, C, H, W) -> (N, C): mean over the spatial dims."""
    arr = np.asarray(features, dtype="float64")
    if arr.ndim != 4:
        raise ValueError(f"expected (N,C,H,W), got shape {arr.shape}")
    return arr.mean(axis=(2, 3))


def fit_probe(X_train: np.ndarray, y_train: np.ndarray):
    """Fit LogisticRegression(max_iter=1000) on (X, y). Returns None if the
    train labels have fewer than 2 classes (probe undefined)."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y_train).astype(int)
    if len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(max_iter=1000)
    clf.fit(np.asarray(X_train, dtype="float64"), y)
    return clf


def evaluate_probe(probe, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """AUROC of the probe's positive-class scores on the val 0/1 subset.
    Returns NaN if the probe is None or the val labels have <2 classes."""
    from sklearn.metrics import roc_auc_score

    if probe is None:
        return float("nan")
    y = np.asarray(y_val).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    scores = probe.predict_proba(np.asarray(X_val, dtype="float64"))[:, 1]
    return float(roc_auc_score(y, scores))


def extract_features(model, dataset, device: str = "cpu") -> tuple[np.ndarray, list[str]]:
    """Forward each image through ``model.features`` -> ReLU -> global-average
    pool -> 1024-d vector. Returns (X [N,1024], image_paths)."""
    import torch
    import torch.nn.functional as F

    feats: list[np.ndarray] = []
    paths: list[str] = []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample["image"]
            if not torch.is_tensor(image):
                image = torch.as_tensor(np.asarray(image), dtype=torch.float32)
            image = image.float().unsqueeze(0).to(device)        # (1,1,224,224)
            fmap = model.features(image)                          # (1,1024,7,7)
            pooled = F.adaptive_avg_pool2d(F.relu(fmap), (1, 1)).flatten(1)  # (1,1024)
            feats.append(pooled.cpu().numpy()[0])
            paths.append(str(sample["image_path"]))
    return np.vstack(feats), paths


def _labels_for(dataset, label: str) -> np.ndarray:
    """Per-sample ground-truth vector for one label, in dataset order."""
    idx = list(dataset.label_columns).index(label)
    return np.array([float(np.asarray(dataset[i]["labels"])[idx]) for i in range(len(dataset))])


def main():
    ap = argparse.ArgumentParser(
        description="Linear probe: LogReg on frozen DenseNet features, train vs CNN AUROC."
    )
    ap.add_argument("--train-root", type=Path, default=Path("data/chexpert"),
                    help="Dir with metadata_train.csv + PNG_train/ (from `src.data fetch-train`).")
    ap.add_argument("--train-metadata", default="metadata_train.csv")
    ap.add_argument("--train-image-dir", default="PNG_train")
    ap.add_argument("--val-root", type=Path, default=Path("data/chexpert"),
                    help="Dir with metadata.csv + PNG_valid/ (the existing val set).")
    ap.add_argument("--cnn-auroc", type=Path, default=Path("results/auroc.csv"),
                    help="Baseline CNN AUROC CSV for side-by-side comparison.")
    ap.add_argument("--output-dir", type=Path, default=Path("results"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from src.data import (
        CheXpertValidDataset, CHEXPERT_COMPETITION_LABELS,
        CHEXPERT_METADATA_FILENAME, CHEXPERT_IMAGE_DIRNAME,
    )
    from src.models import build_model

    set_seed(229)
    labels = list(CHEXPERT_COMPETITION_LABELS)

    train_ds = CheXpertValidDataset(
        metadata_csv=args.train_root / args.train_metadata,
        image_root=args.train_root / args.train_image_dir,
        label_columns=labels,
    )
    val_ds = CheXpertValidDataset(
        metadata_csv=args.val_root / CHEXPERT_METADATA_FILENAME,
        image_root=args.val_root / CHEXPERT_IMAGE_DIRNAME,
        label_columns=labels,
    )

    model = build_model("densenet121")
    model = model.to(args.device)

    print(f"extracting train features ({len(train_ds)} images)...")
    Xtr, _ = extract_features(model, train_ds, device=args.device)
    print(f"extracting val features ({len(val_ds)} images)...")
    Xval, _ = extract_features(model, val_ds, device=args.device)

    cnn = {}
    if args.cnn_auroc.exists():
        cdf = pd.read_csv(args.cnn_auroc)
        # tolerate either {label,auroc} or a wide single-row layout
        if {"label", "auroc"} <= set(cdf.columns):
            cnn = dict(zip(cdf["label"], cdf["auroc"]))
        else:
            cnn = {c: float(cdf[c].iloc[0]) for c in cdf.columns if c in labels}

    rows = []
    for label in labels:
        ytr = _labels_for(train_ds, label)
        yval = _labels_for(val_ds, label)
        # restrict to scorable {0,1} rows on each side (drop -1/NaN)
        tr_ok = np.isin(ytr, (0.0, 1.0))
        val_ok = np.isin(yval, (0.0, 1.0))
        probe = fit_probe(Xtr[tr_ok], ytr[tr_ok])
        probe_auroc = evaluate_probe(probe, Xval[val_ok], yval[val_ok])
        n_val_pos = int((yval[val_ok] == 1).sum())
        n_val_neg = int((yval[val_ok] == 0).sum())
        rows.append({
            "label": label,
            "cnn_auroc": cnn.get(label, float("nan")),
            "probe_auroc": probe_auroc,
            "n_train_pos": int((ytr[tr_ok] == 1).sum()),
            "n_train_neg": int((ytr[tr_ok] == 0).sum()),
            "n_val_pos": n_val_pos,
            "n_val_neg": n_val_neg,
            "ranked": bool(n_val_pos >= MIN_N and n_val_neg >= MIN_N),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out = args.output_dir / "linear_probe.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
