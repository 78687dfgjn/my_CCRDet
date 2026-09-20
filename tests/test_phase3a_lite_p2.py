import torch
from mmcv import Config
from mmdet.models import build_detector
from mmdet.models.detectors.afdet import Fusion


EXPECTED_SHAPES = [
    (1, 256, 32, 40),
    (1, 256, 16, 20),
    (1, 256, 8, 10),
    (1, 256, 4, 5),
    (1, 256, 2, 3),
]


def test_lite_p2_forward():
    cfg = Config.fromfile('configs_local/exp_p2p6_seed0.py')
    cfg.model.backbone.pretrained = None
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    rgb = torch.randn(1, 3, 128, 160, device=device)
    thermal = torch.randn(1, 3, 128, 160, device=device)
    with torch.no_grad():
        features = model.extract_feat((rgb, thermal))
    shapes = [tuple(feature.shape) for feature in features]
    assert shapes == EXPECTED_SHAPES
    assert model.fusion_types == [
        'fusion_cat', 'fusion', 'fusion', 'fusion', 'fusion_cat']
    return shapes


def test_attention_skip_guard():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer = Fusion(4, attention_mode='skip', attention_hw_threshold=4)
    layer.to(device).eval()
    rgb = torch.randn(1, 4, 3, 2, device=device)
    thermal = torch.randn(1, 4, 3, 2, device=device)
    with torch.no_grad():
        output = layer(rgb, thermal)
    assert torch.equal(output, rgb + thermal)
    return tuple(output.shape)


if __name__ == '__main__':
    print('lite_p2_shapes:', test_lite_p2_forward())
    print('attention_skip_shape:', test_attention_skip_guard())
