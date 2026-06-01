"""Model builders for chest X-ray classifiers."""

from __future__ import annotations

from pathlib import Path

# CheXpert competition label -> TorchXRayVision pathology name.
# xrv exposes pleural effusion as "Effusion"; CheXpert (and our config /
# Andrew's saliency contract) calls it "Pleural Effusion". The other four
# competition labels match xrv's names exactly, so they map to themselves.
XRV_PATHOLOGY_ALIASES = {"Pleural Effusion": "Effusion"}

SUPPORTED_MODELS = {"densenet121"}


def build_model(name: str, num_classes: int | None = None, checkpoint: str | None = None):
    """Return a CheXpert-pretrained classifier in eval mode.

    Only ``densenet121`` is supported: a pretrained TorchXRayVision
    DenseNet-121 is itself a legitimate baseline to report. ``num_classes`` is
    accepted for interface compatibility but ignored — the pretrained model has
    a fixed multi-label head over ``model.pathologies``.

    ``checkpoint`` (optional) loads fine-tuned weights on top of the pretrained
    architecture. When ``None`` (the default), the function is byte-for-byte the
    original pretrained baseline, so every existing caller (baseline / ola /
    masking / linear_probe) is unaffected. The mitigation experiment passes a
    checkpoint to evaluate a fine-tuned model with the same pathology head and
    ``op_threshs``, so OLL / masking / AUROC are computed identically.
    """
    key = name.lower()
    if key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model {name!r}. Milestone baseline supports: "
            f"{sorted(SUPPORTED_MODELS)}."
        )

    import torch
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights="densenet121-res224-chex")

    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        # accept a raw state_dict or a {"model_state_dict": ...} training ckpt
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

    model.eval()
    return model
