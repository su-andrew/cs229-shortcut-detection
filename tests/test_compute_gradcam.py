import inspect

from src import saliency


def test_compute_gradcam_exists_with_tensor_signature():
    assert hasattr(saliency, "compute_gradcam")
    params = list(inspect.signature(saliency.compute_gradcam).parameters)
    # tensor-first signature, NOT a numpy image_224
    assert params[:3] == ["model", "image_tensor", "target_label"]


def test_render_gradcam_delegates_to_compute_gradcam(monkeypatch):
    import numpy as np

    calls = {}

    def fake_compute(model, image_tensor, target_label, model_pathologies=None):
        calls["hit"] = True
        return np.zeros((224, 224), dtype="float32")

    monkeypatch.setattr(saliency, "compute_gradcam", fake_compute)

    class M:
        pathologies = ["Cardiomegaly"]
    import torch
    saliency.render_gradcam(
        M(), torch.zeros(1, 1, 224, 224), "Cardiomegaly",
        __import__("pathlib").Path("results/_tmp_cam.png"),
        model_pathologies=["Cardiomegaly"],
    )
    assert calls.get("hit") is True
