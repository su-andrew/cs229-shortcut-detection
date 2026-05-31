"""R3: masked-region sensitivity. Heuristic non-anatomical masks -> masked
inference -> per-disease delta-AUROC. A sensitivity probe over predefined
regions, NOT a proof the masked region is the learned shortcut.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.shortcuts import shortcut_reliance_score
from src.utils import set_seed

MIN_N = 10


def make_shortcut_mask(kind: str, shape=(224, 224), frac: float = 0.15) -> np.ndarray:
    """Boolean mask over a predefined non-anatomical region."""
    if kind in ("border", "corners") and frac > 0.5:
        raise ValueError(
            f"frac={frac} > 0.5 masks the entire image for kind={kind!r}"
        )
    h, w = shape
    m = np.zeros(shape, dtype=bool)
    bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
    if kind == "corners":
        m[:bh, :bw] = True; m[:bh, -bw:] = True
        m[-bh:, :bw] = True; m[-bh:, -bw:] = True
    elif kind == "border":
        m[:bh, :] = True; m[-bh:, :] = True
        m[:, :bw] = True; m[:, -bw:] = True
    elif kind == "laterality":  # top-left / top-right L/R marker zones
        m[:bh, :bw] = True; m[:bh, -bw:] = True
    else:
        raise ValueError(f"unknown mask kind: {kind!r}")
    return m


def apply_mask(image_224, mask, fill: str = "mean"):
    """Return a copy of a (1,224,224) image with masked pixels filled."""
    import torch

    if fill not in ("mean", "zero"):
        raise ValueError(f"unknown fill: {fill!r} (expected 'mean' or 'zero')")
    is_tensor = torch.is_tensor(image_224)
    arr = image_224.detach().cpu().numpy() if is_tensor else np.asarray(image_224)
    arr = arr.copy()
    plane = arr[0]  # view into arr; in-place edits below mutate arr
    plane[mask] = float(plane.mean()) if fill == "mean" else 0.0
    if is_tensor:
        return torch.as_tensor(arr, dtype=torch.float32)
    return arr.astype(np.float32)


def rank_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Order results by ΔAUROC, but keep gate-passing (ranked) diseases above
    unranked ones so a degenerate/low-n disease never tops the table on a noisy
    ΔAUROC."""
    return df.sort_values(["ranked", "delta_auroc"], ascending=[False, False])


def delta_auroc(y_true, clean_pred, masked_pred) -> float:
    """AUROC(clean) - AUROC(masked) on the clean 0/1 subset."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    c_all = np.asarray(clean_pred, dtype="float64")
    m_all = np.asarray(masked_pred, dtype="float64")
    valid = np.isin(y, (0, 1)) & np.isfinite(c_all) & np.isfinite(m_all)
    y = y[valid].astype(int)
    c = c_all[valid]; m = m_all[valid]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, c) - roc_auc_score(y, m))


def main():
    ap = argparse.ArgumentParser(description="R3: masked-region sensitivity (delta-AUROC).")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--kind", default="corners", choices=["corners", "border", "laterality"])
    ap.add_argument("--fill", default="mean", choices=["mean", "zero"])
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--num-labels", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    from src.baseline import load_config, _map_labels_to_indices, predict_dataset
    from src.data import CheXpertValidDataset
    from src.models import build_model
    from src.saliency import compute_label_aurocs, lowest_auroc_labels

    set_seed(229)
    cfg = load_config(args.config)
    labels = list(cfg["data"]["label_columns"])
    model = build_model(cfg["model"]["name"]); model.to(args.device).eval()
    idx = _map_labels_to_indices(labels, list(model.pathologies))

    if args.data_root is not None:
        root = Path(args.data_root)
        ds = CheXpertValidDataset(metadata_csv=root / "metadata.csv",
                                  image_root=root / "PNG_valid", label_columns=labels)
    else:
        ds = CheXpertValidDataset(label_columns=labels)

    mask = make_shortcut_mask(args.kind, frac=args.frac)
    clean = predict_dataset(model, ds, labels, idx, device=args.device)
    masked = predict_dataset(model, ds, labels, idx, device=args.device,
                             image_transform=lambda im: apply_mask(im, mask, fill=args.fill))

    audit = lowest_auroc_labels(compute_label_aurocs(clean), args.num_labels)
    rows = []
    for label in labels:
        yt = clean[f"{label}_true"].to_numpy()
        cp = clean[f"{label}_pred"].to_numpy(); mp = masked[f"{label}_pred"].to_numpy()
        valid = np.isin(yt, (0.0, 1.0))
        n_pos = int((yt[valid] == 1).sum()); n_neg = int((yt[valid] == 0).sum())
        rows.append({
            "label": label, "mask_kind": args.kind, "fill": args.fill,
            "delta_auroc": delta_auroc(yt, cp, mp),
            "srs": shortcut_reliance_score(cp[valid], mp[valid]),
            "n_pos": n_pos, "n_neg": n_neg,
            "ranked": bool(n_pos >= MIN_N and n_neg >= MIN_N),
            "audited": label in audit,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = rank_sort(pd.DataFrame(rows))
    out = args.output_dir / "masked_sensitivity.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
