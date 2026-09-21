import json
import os
import sys

import torch
from mmcv import Config

from mmdet.models import build_detector


def build_from(path, enabled):
    cfg = Config.fromfile(path)
    cfg.model.backbone.pretrained = None
    cfg.model.get('pretrained', None)
    cfg.model.pop('pretrained', None)
    cfg.model.pop('init_cfg', None)
    if not enabled:
        cfg.model.pop('p2_detail', None)
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    checkpoint = torch.load(
        '/root/CCRDet/work_dirs/baseline_seed_0/epoch_12.pth',
        map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    incompat = model.load_state_dict(state_dict, strict=False)
    return model.cuda().eval(), list(incompat.missing_keys), list(incompat.unexpected_keys)


def main():
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    base_path = 'configs_local/gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py'
    v2_path = 'configs_local/exp_proxy_spdi_v2_seed0.py'
    base, base_missing, base_unexpected = build_from(base_path, False)
    v2, v2_missing, v2_unexpected = build_from(v2_path, True)
    rgb = torch.randn(1, 3, 512, 640, device='cuda')
    thermal = torch.randn(1, 3, 512, 640, device='cuda')
    with torch.no_grad():
        base_features = base.extract_feat((rgb, thermal))
        v2_features = v2.extract_feat((rgb, thermal))
    rows = []
    for idx, (a, b) in enumerate(zip(base_features, v2_features)):
        diff = (a - b).abs().max().item()
        rows.append(dict(level='P{}'.format(idx + 3),
                         shape=list(a.shape), torch_equal=bool(torch.equal(a, b)),
                         max_abs_diff=float(diff)))
    module = v2.p2_detail_injection
    result = dict(
        checkpoint='/root/CCRDet/work_dirs/baseline_seed_0/epoch_12.pth',
        base_missing_keys=base_missing,
        base_unexpected_keys=base_unexpected,
        v2_missing_keys=v2_missing,
        v2_unexpected_keys=v2_unexpected,
        v2_init_mode=module.init_mode,
        v2_alpha=float(module.alpha.detach().item()),
        v2_pointwise_weight_abs_sum=float(module.pointwise.weight.detach().abs().sum().item()),
        v2_pointwise_bias_abs_sum=float(module.pointwise.bias.detach().abs().sum().item()),
        features=rows,
        identity_pass=all(r['torch_equal'] and r['max_abs_diff'] == 0.0 for r in rows),
    )
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/phase3e_spdi_v2_identity.json', 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result['identity_pass']:
        raise SystemExit('SPDI-v2 identity test FAILED')


if __name__ == '__main__':
    main()
