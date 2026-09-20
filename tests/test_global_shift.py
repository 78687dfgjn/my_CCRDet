import torch

from mmdet.models.utils.global_shift import GlobalThermalShift


def test_off_is_identity():
    x = torch.randn(2, 1, 7, 9)
    y = GlobalThermalShift(mode='off')(x)
    assert y is x
    assert torch.equal(y, x)


def test_fixed_zero_is_identity():
    x = torch.randn(1, 2, 7, 9)
    y = GlobalThermalShift(mode='fixed', dx=0, dy=0)(x)
    assert torch.equal(y, x)


def test_negative_dx_moves_content_left():
    x = torch.zeros(1, 1, 5, 7)
    x[0, 0, 2, 3] = 1.0
    y = GlobalThermalShift(mode='fixed', dx=-1, dy=0)(x)
    assert y[0, 0, 2, 2].item() == 1.0
    assert y.sum().item() == 1.0


def test_learnable_shift_has_gradient():
    x = torch.randn(1, 1, 9, 11)
    module = GlobalThermalShift(mode='learnable', max_shift=2)
    weights = torch.arange(x.numel(), dtype=x.dtype).view_as(x)
    loss = (module(x) * weights).sum()
    loss.backward()
    assert module.raw_shift.grad is not None
    assert torch.isfinite(module.raw_shift.grad).all()
    assert module.raw_shift.grad.abs().sum().item() > 0
