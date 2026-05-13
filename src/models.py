"""Model placeholders for chest X-ray classifiers."""

from __future__ import annotations


def build_model(name: str, num_classes: int):
    """Placeholder for creating ResNet-50 or DenseNet-121 models."""
    raise NotImplementedError(
        f"Add the {name} implementation here for {num_classes} output labels."
    )
