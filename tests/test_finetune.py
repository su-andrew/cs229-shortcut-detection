import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.finetune import masked_bce_loss, maybe_mask_batch


def test_masked_bce_ignores_uncertain_and_nan():
    # 1 example, 3 pathology logits; competition labels map to cols [0,2]
    logits = torch.zeros(1, 3)
    # targets for the 2 selected labels: one valid (1), one uncertain (-1)
    tgts = torch.tensor([[1.0, -1.0]])
    loss = masked_bce_loss(logits, tgts, [0, 2])
    # only the valid entry contributes; logit 0 -> sigmoid .5 -> -log(.5)=ln2
    assert abs(float(loss) - np.log(2)) < 1e-5


def test_masked_bce_all_unsupervised_is_zero_grad_safe():
    logits = torch.zeros(1, 3, requires_grad=True)
    tgts = torch.tensor([[float("nan"), -1.0]])
    loss = masked_bce_loss(logits, tgts, [0, 2])
    assert float(loss) == 0.0
    loss.backward()  # must not error


def test_maybe_mask_batch_zero_frac_is_identity():
    imgs = torch.randn(4, 1, 8, 8)
    mask = np.zeros((8, 8), dtype=bool); mask[0, :] = True
    out = maybe_mask_batch(imgs, mask, frac=0.0, rng=np.random.default_rng(0))
    assert torch.equal(out, imgs)


def test_maybe_mask_batch_full_frac_fills_region_with_mean():
    imgs = torch.arange(1 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
    mask = np.zeros((4, 4), dtype=bool); mask[0, :] = True  # top row
    expected_mean = float(imgs[0, 0].mean())
    out = maybe_mask_batch(imgs, mask, frac=1.0, rng=np.random.default_rng(0))
    assert torch.allclose(out[0, 0][torch.as_tensor(mask)],
                          torch.full((4,), expected_mean))
    # input not mutated
    assert not torch.equal(out, imgs)
