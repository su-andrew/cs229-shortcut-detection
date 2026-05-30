"""Lung-field segmentation (R1): per-image 224x224 lung mask via xrv PSPNet.

The segmenter (chestx_det.PSPNet) runs at 512x512 with its own xrv-normalized
input. The classifier and Grad-CAM run at 224x224. lung_mask co-registers the
two grids (verified visually in the R1 diagnostic) so attribution_outside_mask
compares a CAM and a mask on the SAME grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LUNG_TARGETS = ("Left Lung", "Right Lung")


def resample_to_224(arr: np.ndarray) -> np.ndarray:
    """Bilinear-resample a 2D array to (224, 224) float32."""
    import torch
    import torch.nn.functional as F

    t = torch.as_tensor(np.asarray(arr), dtype=torch.float32)[None, None]
    out = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    return out[0, 0].numpy().astype("float32")


def binarize_lung_logits(logits_224: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Sigmoid + threshold a lung-logit map to a boolean mask."""
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits_224, dtype="float64")))
    return probs > threshold


def load_lung_segmenter(device: str = "cpu"):
    """Load the xrv ChestX-Det PSPNet anatomical segmenter."""
    try:
        import torch
        import torchxrayvision as xrv
    except ModuleNotFoundError as exc:  # pragma: no cover - env guard
        raise RuntimeError(
            "Lung segmentation needs torch and torchxrayvision installed."
        ) from exc

    model = xrv.baseline_models.chestx_det.PSPNet()
    model = model.to(torch.device(device))
    model.eval()
    return model


def lung_mask(seg_model, image_224, device: str = "cpu", threshold: float = 0.5) -> np.ndarray:
    """Return a (224,224) boolean lung-field mask co-registered to the 224 grid.

    image_224: a (1,224,224) xrv-normalized array/tensor (same preprocessing as
    the classifier input — verified in the R1 co-registration diagnostic).
    """
    import torch

    t = torch.as_tensor(np.asarray(image_224), dtype=torch.float32)
    if t.ndim == 3:
        t = t.unsqueeze(0)  # (1,1,224,224)
    t = t.to(torch.device(device))
    with torch.no_grad():
        seg_out = seg_model(t)  # (1,14,512,512)
    targets = list(seg_model.targets)
    idx = [targets.index(name) for name in LUNG_TARGETS]
    lungs_512 = seg_out[0, idx].sum(dim=0).detach().cpu().numpy()  # (512,512) logits
    lungs_224 = resample_to_224(lungs_512)
    return binarize_lung_logits(lungs_224, threshold=threshold)


def overlay_mask(image_224, mask, output_path: Path) -> None:
    """Save a QA overlay of a boolean mask on a 224 image (appendix figure)."""
    import matplotlib.pyplot as plt

    base = np.asarray(image_224, dtype="float32")
    base = base[0] if base.ndim == 3 else base
    base = (base - base.min()) / (np.ptp(base) + 1e-8)
    rgb = np.stack([base, base, np.clip(base + 0.4 * mask, 0, 1)], axis=-1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, np.clip(rgb, 0, 1))
