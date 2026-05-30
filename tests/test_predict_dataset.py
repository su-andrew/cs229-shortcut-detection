import numpy as np
import pytest
import torch

from src.baseline import predict_dataset


class TinyDataset:
    def __init__(self): self._n = 3
    def __len__(self): return self._n
    def __getitem__(self, i):
        return {"image": torch.zeros(1, 224, 224), "labels": np.array([1.0, 0.0]),
                "image_path": f"img{i}.png"}


class TinyModel(torch.nn.Module):
    pathologies = ["Cardiomegaly", "Effusion"]
    def forward(self, x):
        b = x.shape[0]
        return torch.tensor([[0.8, 0.2]] * b)


def test_predict_dataset_builds_pred_columns_and_applies_transform():
    labels = ["Cardiomegaly", "Pleural Effusion"]
    idx = {"Cardiomegaly": 0, "Pleural Effusion": 1}
    seen = {"n": 0}
    def tx(img):
        seen["n"] += 1
        return img
    df = predict_dataset(TinyModel(), TinyDataset(), labels, idx, device="cpu", image_transform=tx)
    assert len(df) == 3
    assert seen["n"] == 3
    assert {"path", "Cardiomegaly_true", "Cardiomegaly_pred",
            "Pleural Effusion_true", "Pleural Effusion_pred"} <= set(df.columns)
    assert df["Cardiomegaly_pred"].iloc[0] == pytest.approx(0.8, abs=1e-5)
