import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchxrayvision")

from src.models import build_model


def test_build_model_default_is_pretrained_eval():
    # default path unchanged: no checkpoint -> pretrained, eval mode
    m = build_model("densenet121")
    assert not m.training
    assert hasattr(m, "pathologies")


def test_build_model_loads_checkpoint(tmp_path):
    # a checkpoint (plain state_dict OR {"model_state_dict": ...}) is loaded
    base = build_model("densenet121")
    # perturb one parameter so we can detect that the checkpoint was applied.
    # Pick a finite-valued tensor (op_threshs holds NaN for blank pathology
    # slots, which breaks an equality check), e.g. the classifier weight.
    import torch as _t
    sd = base.state_dict()
    key = "classifier.weight"
    assert key in sd and _t.isfinite(sd[key]).all()
    sd[key] = sd[key] + 1.0
    ckpt = tmp_path / "ft.pt"
    _t.save({"model_state_dict": sd}, ckpt)

    loaded = build_model("densenet121", checkpoint=str(ckpt))
    assert _t.allclose(loaded.state_dict()[key], sd[key])
    assert not loaded.training  # still returned in eval mode


def test_build_model_missing_checkpoint_raises(tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        build_model("densenet121", checkpoint=str(tmp_path / "nope.pt"))
