"""Masked-augmentation fine-tune (mitigation experiment).

Fine-tunes the CheXpert-pretrained DenseNet-121 on the train sample and saves a
checkpoint. With ``--mask-frac > 0`` it masks a shortcut region (default the
border, the one R3 flagged) on that fraction of training images, so the model
cannot lean on the border and is pushed toward lung-field features. With
``--mask-frac 0`` it is a plain fine-tune, which we train as the CONTROL so any
change in the OLL / masking screens can be attributed to the masking rather than
to fine-tuning per se.

Evaluation is not done here: after training, re-run baseline / ola / masking with
``--checkpoint <path>`` (build_model loads it) and compare pretrained vs control
vs masked. The classifier head and op_threshs are unchanged, so AUROC, OLL, and
masked-region sensitivity are all computed identically to the pretrained run.

Training uses raw logits + a masked BCEWithLogitsLoss over the five competition
labels: only entries that are genuinely labeled (0/1) contribute to the loss;
uncertain (-1) and unmentioned (NaN) entries are masked out per example. This is
a full fine-tune (backbone + head) at a low LR -- freezing the backbone would
leave Grad-CAM attribution unchanged and defeat the mitigation.

Designed to run on a Colab GPU (``--device cuda``); also runs on mps/cpu.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def masked_bce_loss(logits, targets, label_indices):
    """BCEWithLogits over the competition-label columns, ignoring non-{0,1}.

    logits: (B, n_pathologies) raw model outputs.
    targets: (B, 5) ground truth in {0,1,-1,NaN} for the 5 competition labels.
    label_indices: list[int] mapping each of the 5 labels to its logit column.
    Returns a scalar loss averaged over supervised (0/1) entries only.
    """
    import torch
    import torch.nn.functional as F

    sel = logits[:, label_indices]                      # (B, 5)
    valid = (targets == 0) | (targets == 1)             # bool (B, 5); drops -1/NaN
    if valid.sum() == 0:
        return sel.sum() * 0.0                          # no supervision this batch
    per = F.binary_cross_entropy_with_logits(
        sel, torch.nan_to_num(targets, nan=0.0).float(), reduction="none"
    )
    return per[valid].mean()


def maybe_mask_batch(images, mask_bool, frac, rng):
    """Apply a boolean region mask (mean-fill) to a random ``frac`` of a batch.

    images: (B,1,H,W) tensor. mask_bool: (H,W) numpy bool. Returns a new tensor.
    Per-image mean fill matches src.masking.apply_mask semantics.
    """
    import torch

    if frac <= 0:
        return images
    out = images.clone()
    m = torch.as_tensor(mask_bool, dtype=torch.bool, device=images.device)
    for b in range(out.shape[0]):
        if rng.random() < frac:
            plane = out[b, 0]
            plane[m] = plane.mean()
    return out


def train(
    *,
    data_root: Path,
    train_metadata: str,
    train_image_dir: str,
    out_path: Path,
    mask_kind: str = "border",
    mask_frac: float = 0.5,
    frac_region: float = 0.15,
    epochs: int = 4,
    lr: float = 1e-4,
    batch_size: int = 16,
    device_str: str = "auto",
    seed: int = 229,
):
    import torch
    from torch.utils.data import DataLoader

    from src.data import CheXpertValidDataset, CHEXPERT_COMPETITION_LABELS
    from src.masking import make_shortcut_mask
    from src.models import build_model
    from src.baseline import _map_labels_to_indices, select_device
    from src.utils import set_seed

    set_seed(seed)
    device = select_device(device_str)
    labels = list(CHEXPERT_COMPETITION_LABELS)

    ds = CheXpertValidDataset(
        metadata_csv=data_root / train_metadata,
        image_root=data_root / train_image_dir,
        label_columns=labels,
    )
    if len(ds) == 0:
        raise RuntimeError(f"No train images under {data_root/train_image_dir}.")

    def collate(samples):
        imgs = torch.stack([
            s["image"] if torch.is_tensor(s["image"])
            else torch.as_tensor(np.asarray(s["image"]), dtype=torch.float32)
            for s in samples
        ]).float()
        tgts = torch.as_tensor(
            np.stack([np.asarray(s["labels"], dtype="float32") for s in samples])
        )
        return imgs, tgts

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    model = build_model("densenet121").to(device)
    model.train()
    label_idx = _map_labels_to_indices(labels, list(model.pathologies))
    idx_list = [label_idx[l] for l in labels]

    mask_bool = make_shortcut_mask(mask_kind, frac=frac_region) if mask_frac > 0 else None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        running, nb = 0.0, 0
        for imgs, tgts in loader:
            imgs = imgs.to(device)
            tgts = tgts.to(device)
            if mask_bool is not None:
                imgs = maybe_mask_batch(imgs, mask_bool, mask_frac, rng)
            opt.zero_grad()
            logits = _raw_logits(model, imgs)
            loss = masked_bce_loss(logits, tgts, idx_list)
            loss.backward()
            opt.step()
            running += float(loss.detach()); nb += 1
        print(f"epoch {epoch}/{epochs}  mean_loss={running/max(nb,1):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.save({"model_state_dict": model.state_dict(),
                "mask_kind": mask_kind, "mask_frac": mask_frac,
                "epochs": epochs, "lr": lr, "seed": seed}, str(out_path))
    print(f"saved checkpoint -> {out_path}")
    return out_path


def _raw_logits(model, imgs):
    """Raw classifier logits, bypassing op_threshs (xrv forward applies it)."""
    import torch
    import torch.nn.functional as F

    feats = model.features(imgs)
    out = F.relu(feats)
    out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
    return model.classifier(out)


def main():
    ap = argparse.ArgumentParser(description="Masked-augmentation fine-tune (mitigation).")
    ap.add_argument("--data-root", type=Path, default=Path("data/chexpert"))
    ap.add_argument("--train-metadata", default="metadata_train.csv")
    ap.add_argument("--train-image-dir", default="PNG_train")
    ap.add_argument("--out", type=Path, required=True,
                    help="Checkpoint output path, e.g. checkpoints/masked.pt")
    ap.add_argument("--mask-kind", default="border", choices=["corners", "border", "laterality"])
    ap.add_argument("--mask-frac", type=float, default=0.5,
                    help="Fraction of train images to mask. 0.0 = control (plain fine-tune).")
    ap.add_argument("--frac-region", type=float, default=0.15, help="Mask region size.")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=229)
    args = ap.parse_args()

    train(
        data_root=args.data_root, train_metadata=args.train_metadata,
        train_image_dir=args.train_image_dir, out_path=args.out,
        mask_kind=args.mask_kind, mask_frac=args.mask_frac,
        frac_region=args.frac_region, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device_str=args.device, seed=args.seed,
    )


if __name__ == "__main__":
    main()
