"""Mitigation evaluation: re-run the screens on a fine-tuned checkpoint.

Given a checkpoint (or none = pretrained), compute per-disease validation AUROC
and the R2 OLL screen, writing tagged CSVs so pretrained / control / masked can
be compared side by side. (R3 masked-region sensitivity is run via
``python -m src.masking --checkpoint ...`` once that flag is wired; this helper
covers the AUROC + OLL half, which is what the mitigation claim rests on.)

Usage (after training on Colab and downloading the checkpoints):
  python -m src.mitigation_eval --tag pretrained
  python -m src.mitigation_eval --tag control --checkpoint checkpoints/control.pt
  python -m src.mitigation_eval --tag masked  --checkpoint checkpoints/masked.pt

Outputs results/mitigation_{tag}_auroc.csv and results/mitigation_{tag}_oll.csv.
The mitigation story: vs pretrained, the MASKED model should show lower OLL
(less out-of-lung attention) while AUROC holds; the CONTROL isolates whether any
change is due to masking rather than fine-tuning alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on AUROC + OLL.")
    ap.add_argument("--tag", required=True, help="pretrained | control | masked")
    ap.add_argument("--checkpoint", default=None, help="None -> pretrained baseline.")
    ap.add_argument("--data-root", type=Path, default=Path("data/chexpert"))
    ap.add_argument("--output-dir", type=Path, default=Path("results"))
    ap.add_argument("--num-labels", type=int, default=3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from sklearn.metrics import roc_auc_score

    from src.baseline import (
        select_device, _map_labels_to_indices, predict_dataset,
    )
    from src.data import CheXpertValidDataset, CHEXPERT_COMPETITION_LABELS
    from src.models import build_model
    from src.ola import compute_oll_rows, summarize_label
    from src.saliency import compute_label_aurocs, lowest_auroc_labels
    from src.segmentation import load_lung_segmenter
    from src.utils import set_seed

    set_seed(229)
    device = select_device(args.device)
    labels = list(CHEXPERT_COMPETITION_LABELS)

    model = build_model("densenet121", checkpoint=args.checkpoint).to(device)
    label_to_index = _map_labels_to_indices(labels, list(model.pathologies))

    ds = CheXpertValidDataset(
        metadata_csv=args.data_root / "metadata.csv",
        image_root=args.data_root / "PNG_valid",
        label_columns=labels,
    )

    # --- AUROC ---
    preds = predict_dataset(model, ds, labels, label_to_index, device=device)
    rows = []
    for label in labels:
        yt = preds[f"{label}_true"].to_numpy()
        yp = preds[f"{label}_pred"].to_numpy()
        import numpy as np
        valid = np.isin(yt, (0.0, 1.0))
        y = yt[valid].astype(int)
        auroc = float(roc_auc_score(y, yp[valid])) if len(set(y)) > 1 else float("nan")
        rows.append({"label": label, "auroc": auroc,
                     "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    auroc_df = pd.DataFrame(rows)
    auroc_out = args.output_dir / f"mitigation_{args.tag}_auroc.csv"
    auroc_df.to_csv(auroc_out, index=False)
    print(f"[{args.tag}] AUROC:"); print(auroc_df.to_string(index=False))

    # --- OLL (reuse the R2 pipeline on this model) ---
    seg = load_lung_segmenter(device=device if device != "mps" else "cpu")
    audit = lowest_auroc_labels(compute_label_aurocs(preds), args.num_labels)
    oll_rows = []
    for label in audit:
        per_img, n_empty = compute_oll_rows(
            preds, label, args.data_root, seg, model, device=device,
        )
        oll_rows.append(summarize_label(label, per_img, n_empty_mask=n_empty))
    oll_df = pd.DataFrame(oll_rows)
    oll_out = args.output_dir / f"mitigation_{args.tag}_oll.csv"
    oll_df.to_csv(oll_out, index=False)
    print(f"[{args.tag}] OLL:"); print(oll_df.to_string(index=False))
    print(f"wrote {auroc_out} and {oll_out}")


if __name__ == "__main__":
    main()
