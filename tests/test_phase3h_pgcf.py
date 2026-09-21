"""Identity and two-step gradient tests for Phase 3H PGCF."""
import json
from pathlib import Path

import torch
from mmcv import Config
from mmdet.models import build_detector


SOURCE = '/hy-tmp/CCRDet_assets/phase3a_semantic_warmstart_seed0.pth'
BASE_CONFIG = 'configs_local/exp_p2p6_seed0.py'
PGCF_CONFIG = 'configs_local/exp_proxy_pgcf_p2p6_seed0.py'
ALLOWED_MISSING = (
    'fuse.0.reduce.',
    'fuse.0.depthwise.',
    'fuse.0.gate_logits.',
)


def build_loaded(config_path):
    cfg = Config.fromfile(config_path)
    cfg.model.backbone.pretrained = None
    cfg.model.pop('pretrained', None)
    cfg.model.pop('init_cfg', None)
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    source = torch.load(SOURCE, map_location='cpu')['state_dict']
    incompat = model.load_state_dict(source, strict=False)
    return model.cuda().eval(), cfg, list(incompat.missing_keys), list(
        incompat.unexpected_keys)


def load_source_into(model):
    source = torch.load(SOURCE, map_location='cpu')['state_dict']
    result = model.load_state_dict(source, strict=False)
    return source, list(result.missing_keys), list(result.unexpected_keys)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Phase 3H tests')
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    base, _, base_missing, base_unexpected = build_loaded(BASE_CONFIG)
    candidate, cfg, candidate_missing, candidate_unexpected = build_loaded(
        PGCF_CONFIG)

    missing_ok = all(
        any(key.startswith(prefix) for prefix in ALLOWED_MISSING)
        for key in candidate_missing)
    compatibility_pass = (
        not base_missing and not base_unexpected and
        not candidate_unexpected and missing_ok and
        bool(candidate_missing))
    module_types = [type(module).__name__ for module in candidate.fuse]

    rgb = torch.randn(1, 3, 128, 160, device='cuda')
    thermal = torch.randn(1, 3, 128, 160, device='cuda')
    with torch.no_grad():
        base_features = base.extract_feat((rgb, thermal))
        candidate_features = candidate.extract_feat((rgb, thermal))

    feature_rows = []
    for index, (left, right) in enumerate(
            zip(base_features, candidate_features)):
        feature_rows.append({
            'level': 'P{}'.format(index + 2),
            'shape': list(left.shape),
            'torch_equal': bool(torch.equal(left, right)),
            'max_abs_diff': float((left - right).abs().max().item()),
        })
    identity_pass = all(row['torch_equal'] for row in feature_rows)

    pgcf = candidate.fuse[0]
    initial_gate_weight_norm = float(
        pgcf.gate_logits.weight.detach().norm().item())
    initial_gate_bias_norm = float(
        pgcf.gate_logits.bias.detach().norm().item())
    pgcf.train()
    optimizer = torch.optim.SGD(pgcf.parameters(), lr=0.001)
    gradient_rows = []
    for iteration in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        x_rgb = torch.randn(2, 256, 32, 40, device='cuda',
                             requires_grad=True)
        x_thermal = torch.randn(2, 256, 32, 40, device='cuda',
                                  requires_grad=True)
        output = pgcf(x_rgb, x_thermal)
        loss = (output * output).mean() + 0.01 * output.mean()
        loss.backward()

        def norm(name):
            value = getattr(pgcf, name).weight.grad
            return float(value.detach().norm().item()) if value is not None else 0.0

        gradient_rows.append({
            'iteration': iteration,
            'loss': float(loss.detach().item()),
            'gate_logits_grad_norm': norm('gate_logits'),
            'reduce_grad_norm': norm('reduce'),
            'depthwise_grad_norm': norm('depthwise'),
            'conv1x1_grad_norm': norm('conv1x1'),
            'gate_logits_weight_norm_before_step': float(
                pgcf.gate_logits.weight.detach().norm().item()),
            'reduce_weight_norm': float(pgcf.reduce.weight.detach().norm().item()),
            'depthwise_weight_norm': float(
                pgcf.depthwise.weight.detach().norm().item()),
        })
        if not all(torch.isfinite(parameter).all()
                   for parameter in pgcf.parameters()):
            raise RuntimeError('non-finite PGCF parameter')
        optimizer.step()

    first, second = gradient_rows
    gradient_pass = (
        first['gate_logits_grad_norm'] > 0.0 and
        first['conv1x1_grad_norm'] > 0.0 and
        second['gate_logits_grad_norm'] > 0.0 and
        second['reduce_grad_norm'] > 0.0 and
        second['depthwise_grad_norm'] > 0.0 and
        all(bool(torch.isfinite(
            torch.tensor(list(row.values())[1:])).all().item())
            for row in gradient_rows))

    identity = {
        'status': 'PASS' if identity_pass else 'FAIL',
        'source_checkpoint': SOURCE,
        'config_a': BASE_CONFIG,
        'config_b': PGCF_CONFIG,
        'module_types_b': module_types,
        'base_missing_keys': base_missing,
        'base_unexpected_keys': base_unexpected,
        'candidate_missing_keys': candidate_missing,
        'candidate_unexpected_keys': candidate_unexpected,
        'allowed_new_key_prefixes': list(ALLOWED_MISSING),
        'warmstart_compatibility_pass': compatibility_pass,
        'features': feature_rows,
        'initial_gate_logit_weight_norm': initial_gate_weight_norm,
        'initial_gate_logit_bias_norm': initial_gate_bias_norm,
        'identity_pass': identity_pass,
    }
    gradient = {
        'status': 'PASS' if gradient_pass else 'FAIL',
        'initialization': 'gate_logits_weight_and_bias_zero',
        'iterations': gradient_rows,
        'iteration_1_reduce_and_depthwise_zero_allowed': True,
        'iteration_2_reduce_and_depthwise_nonzero_required': True,
        'gradient_pass': gradient_pass,
    }
    Path('experiments').mkdir(exist_ok=True)
    Path('experiments/phase3h_identity.json').write_text(
        json.dumps(identity, indent=2))
    Path('experiments/phase3h_gradient.json').write_text(
        json.dumps(gradient, indent=2))
    print(json.dumps({'identity': identity, 'gradient': gradient}, indent=2))
    if not identity_pass or not compatibility_pass or not gradient_pass:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
