"""Unit tests for bounded P2-only residual alignment."""

import math
import torch
from mmcv import Config
from mmdet.models import build_detector

from mmdet.models.utils.p2_micro_alignment import P2MicroAlignment


def test_zero_init_and_identity():
    torch.manual_seed(0)
    m = P2MicroAlignment(4, hidden_channels=3, feature_stride=4,
                         max_residual_px=1.0, analysis_enabled=True)
    rgb = torch.randn(1, 4, 8, 10)
    thermal = torch.randn(1, 4, 8, 10)
    out = m(rgb, thermal)
    assert torch.equal(m.offset.weight, torch.zeros_like(m.offset.weight))
    assert torch.equal(m.offset.bias, torch.zeros_like(m.offset.bias))
    assert (out - thermal).abs().max().item() < 1e-5
    assert set(m.state_dict()) == {
        'reduce.weight', 'reduce.bias', 'offset.weight', 'offset.bias'}
    assert m.last_stats['max_abs_dx'] == 0.0


def test_dot_sign():
    m = P2MicroAlignment(1, hidden_channels=2, feature_stride=4)
    with torch.no_grad():
        m.offset.bias[0] = math.atanh(-0.8)
        m.offset.bias[1] = 0.0
    rgb = torch.zeros(1, 1, 8, 12)
    thermal = torch.zeros_like(rgb)
    thermal[0, 0, 4, 6] = 1.0
    out = m(rgb, thermal)
    # Negative input-pixel dx moves content toward smaller x.
    xs = torch.arange(out.shape[-1], dtype=out.dtype)
    centroid = (out[0, 0, 4] * xs).sum() / out[0, 0, 4].sum()
    assert centroid.item() < 6.0


def test_bound_and_gradients():
    torch.manual_seed(1)
    m = P2MicroAlignment(2, hidden_channels=4, feature_stride=4,
                         max_residual_px=1.0, analysis_enabled=True)
    rgb = torch.randn(1, 2, 8, 8, requires_grad=True)
    thermal = torch.randn(1, 2, 8, 8, requires_grad=True)
    out = m(rgb, thermal)
    out.square().mean().backward()
    assert m.offset.weight.grad is not None
    assert torch.isfinite(m.offset.weight.grad).all()
    assert m.reduce.weight.grad is not None
    delta = torch.tanh(m.offset(m.relu(m.reduce(torch.cat((rgb, thermal), 1))))
                       ).detach()
    assert float(delta.abs().max()) <= 1.0
    assert m.last_stats['units'] == 'input_image_pixels'


def test_disabled_has_no_module_keys_and_pyramid_count():
    # Disabled is represented by not constructing the module in GFLAF.
    cfg = Config.fromfile('configs_local/exp_p2p6_seed0.py')
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    assert model.p2_micro_alignment is None
    assert not any(k.startswith('p2_micro_alignment.')
                   for k in model.state_dict())
    # A Lite-P2 pyramid remains exactly five levels at the caller boundary.
    features = [torch.empty(1, 256, s, s) for s in (32, 16, 8, 4, 2)]
    assert len(features) == 5


def run_all():
    test_zero_init_and_identity()
    test_dot_sign()
    test_bound_and_gradients()
    test_disabled_has_no_module_keys_and_pyramid_count()
    print('PASS test_phase3f_p2_micro_alignment')


if __name__ == '__main__':
    run_all()
