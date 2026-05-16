"""Model builders for chest X-ray classifiers."""

from __future__ import annotations

# CheXpert competition label -> TorchXRayVision pathology name.
# xrv exposes pleural effusion as "Effusion"; CheXpert (and our config /
# Andrew's saliency contract) calls it "Pleural Effusion". The other four
# competition labels match xrv's names exactly, so they map to themselves.
XRV_PATHOLOGY_ALIASES = {"Pleural Effusion": "Effusion"}

SUPPORTED_MODELS = {"densenet121"}


def build_model(name: str, num_classes: int | None = None):
    """Return a CheXpert-pretrained classifier in eval mode.

    Only ``densenet121`` is supported for the milestone: a pretrained
    TorchXRayVision DenseNet-121 is itself a legitimate baseline to report,
    and no training happens for the milestone. ``num_classes`` is accepted
    for interface compatibility but ignored — the pretrained model has a
    fixed multi-label head over ``model.pathologies``.
    """
    key = name.lower()
    if key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model {name!r}. Milestone baseline supports: "
            f"{sorted(SUPPORTED_MODELS)}."
        )

    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.eval()
    return model
