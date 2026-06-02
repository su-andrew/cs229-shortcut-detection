"""Exp 3: AP/PA acquisition-confound probe.

Two-step shortcut argument:
  (1) label correlation -- AP (portable/bedside) scans are 2-4x more likely to
      carry a positive disease label than PA (standing) scans in this val set,
      so acquisition protocol is a spurious cue correlated with the targets.
  (2) decodability -- a linear probe on the FROZEN DenseNet penultimate features
      predicts AP-vs-PA well above chance, i.e. the representation encodes that
      confound.

Together these establish a concrete shortcut *mechanism* the saliency screens
may be detecting. This probe does NOT prove the disease head uses the cue
(that needs an intervention); we state it as suggestive, not causal.

Uses stratified k-fold cross-validated AUROC (the 202 AP/PA val images are few,
33 PA) rather than a single split, and reuses extract_features / fit_probe.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import set_seed


def label_correlation(metadata: pd.DataFrame, labels) -> pd.DataFrame:
    """Per-disease positive rate among AP vs PA scorable rows (the step-1 gate)."""
    m = metadata[metadata["ap_pa"].isin(["AP", "PA"])]
    rows = []
    for l in labels:
        sub = m[m[l].isin([0.0, 1.0])]
        ap = sub[sub["ap_pa"] == "AP"][l]
        pa = sub[sub["ap_pa"] == "PA"][l]
        rows.append({
            "label": l,
            "ap_pos_rate": float(ap.mean()) if len(ap) else float("nan"),
            "pa_pos_rate": float(pa.mean()) if len(pa) else float("nan"),
            "n_ap": int(len(ap)), "n_pa": int(len(pa)),
        })
    return pd.DataFrame(rows)


def cv_probe_auroc(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 229):
    """Stratified k-fold CV AUROC for a logistic-regression probe on (X, y).

    Returns (mean_auroc, lo, hi) where lo/hi are the min/max fold AUROC (a
    transparent spread for a small, few-positive sample -- not a bootstrap CI).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"), float("nan"))
    # cap folds so the rare class (PA) has >=1 per fold
    minority = int(min(np.bincount(y)))
    k = max(2, min(n_splits, minority))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(scaler.transform(X[tr]), y[tr])
        s = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], s))
    if not aucs:
        return (float("nan"), float("nan"), float("nan"))
    a = np.asarray(aucs, dtype="float64")
    return (float(a.mean()), float(a.min()), float(a.max()))


def main():
    ap = argparse.ArgumentParser(description="Exp 3: AP/PA acquisition-confound probe.")
    ap.add_argument("--data-root", type=Path, default=Path("data/chexpert"))
    ap.add_argument("--output-dir", type=Path, default=Path("results"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from src.data import CheXpertValidDataset, CHEXPERT_COMPETITION_LABELS
    from src.models import build_model
    from src.linear_probe import extract_features
    from src.baseline import select_device

    set_seed(229)
    device = select_device(args.device)
    labels = list(CHEXPERT_COMPETITION_LABELS)

    meta = pd.read_csv(args.data_root / "metadata.csv")

    # step 1: label correlation (no model needed)
    corr = label_correlation(meta, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corr.to_csv(args.output_dir / "confound_label_correlation.csv", index=False)
    print("=== AP/PA label correlation (step 1) ===")
    print(corr.to_string(index=False))

    # step 2: decodability probe on frozen features
    model = build_model("densenet121").to(device)
    ds = CheXpertValidDataset(
        metadata_csv=args.data_root / "metadata.csv",
        image_root=args.data_root / "PNG_valid",
        label_columns=labels,
    )
    X, paths = extract_features(model, ds, device=device)

    # align ap_pa to feature rows by the trailing patient/study/view key.
    # Dataset paths are absolute (data/chexpert/PNG_valid/patient.../view.png);
    # metadata paths are relative (patient.../view.png). Join on the last three
    # components + extension-stripped filename so both sides match.
    def _key(p):
        parts = Path(str(p)).with_suffix("").parts
        return "/".join(parts[-3:])

    pcol = "path" if "path" in meta.columns else meta.columns[0]
    meta = meta.copy()
    meta["_key"] = meta[pcol].map(_key)
    key_to_appa = dict(zip(meta["_key"], meta["ap_pa"]))
    appa = np.array([key_to_appa.get(_key(p), None) for p in paths], dtype=object)

    keep = np.array([v in ("AP", "PA") for v in appa])
    if keep.sum() == 0:
        raise RuntimeError(
            "AP/PA join produced 0 matches — the metadata path key and the "
            "dataset image_path key did not align (check the _key join / path "
            "structure)."
        )
    Xk = X[keep]
    yk = (appa[keep] == "AP").astype(int)   # 1=AP, 0=PA
    print(f"\n=== AP/PA decodability probe (step 2) ===")
    print(f"usable images: {keep.sum()}  (AP={int(yk.sum())}, PA={int((yk==0).sum())})")

    mean, lo, hi = cv_probe_auroc(Xk, yk)
    print(f"CV probe AUROC (AP vs PA): {mean:.3f}  [folds {lo:.3f}–{hi:.3f}]")
    pd.DataFrame([{"target": "AP_vs_PA", "cv_auroc_mean": mean,
                   "fold_lo": lo, "fold_hi": hi,
                   "n_ap": int(yk.sum()), "n_pa": int((yk == 0).sum())}]
                 ).to_csv(args.output_dir / "confound_probe_auroc.csv", index=False)
    print(f"\nwrote confound_label_correlation.csv + confound_probe_auroc.csv")


if __name__ == "__main__":
    main()
