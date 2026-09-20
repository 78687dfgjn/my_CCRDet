"""Phase 3D-PREP tests and low-cost memory smoke for SPDI.

This script intentionally performs only five train steps for resource testing;
it is not a training run or a paper experiment.
"""

import json
import subprocess
import time
from pathlib import Path

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.apis import set_random_seed
from mmdet.core import build_optimizer
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CHECKPOINT = ROOT / 'weights' / 'weight_rgbt.pth'
REPORT = ROOT / 'experiments' / 'phase3d_spdi_prep.json'


def _cfg(path):
    cfg = Config.fromfile(str(ROOT / path))
    cfg.model.backbone.pretrained = None
    return cfg


def _build(cfg):
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    return model


def _state(model, checkpoint, strict=False):
    state = torch.load(str(checkpoint), map_location='cpu')
    state = state.get('state_dict', state)
    incompatible = model.load_state_dict(state, strict=strict)
    return {
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
        'checkpoint_keys': len(state),
        'model_keys': len(model.state_dict()),
    }


def _max_diff(left, right):
    diffs = [
        float((a - b).abs().max().detach().cpu())
        for a, b in zip(left, right)]
    return max(diffs) if diffs else 0.0


def _finite_nonzero(grad):
    return (
        grad is not None and
        bool(torch.isfinite(grad).all()) and
        float(grad.abs().max().detach().cpu()) > 0.0)


def _driver_used_mb():
    try:
        text = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.STDOUT, text=True)
        return int(text.strip().splitlines()[0])
    except Exception:
        return None


def _parameter_counts(module):
    def count(value):
        return sum(parameter.numel() for parameter in value.parameters())

    counts = {
        'p2_fusion': count(module.p2_fusion),
        'depthwise_conv': count(module.depthwise),
        'pointwise_conv': count(module.pointwise),
        'gate_conv': count(module.gate),
        'alpha': module.alpha.numel(),
    }
    counts['total'] = sum(counts.values())
    return counts


def _flops(module, p2_hw=(128, 160)):
    """Approximate multiply-add FLOPs, counting one MAC as two FLOPs."""
    p2_h, p2_w = p2_hw
    p3_h, p3_w = p2_h // 2, p2_w // 2
    c = module.p2_fusion.in_channels // 2
    out_c = module.p2_fusion.out_channels
    detail_c = module.pointwise.out_channels
    gate_c = module.gate.out_channels
    p2_fusion = 2 * p2_h * p2_w * (2 * c) * out_c
    depthwise = 2 * p3_h * p3_w * detail_c * 3 * 3
    pointwise = 2 * p3_h * p3_w * detail_c * detail_c
    gate = 2 * p3_h * p3_w * (2 * detail_c) * gate_c
    return {
        'p2_fusion': p2_fusion,
        'depthwise_conv': depthwise,
        'pointwise_conv': pointwise,
        'gate_conv': gate,
        'total': p2_fusion + depthwise + pointwise + gate,
        'counting': '2 FLOPs per multiply-add',
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the Phase 3D-PREP test')
    device = torch.device('cuda:0')
    set_random_seed(0, deterministic=True)

    disabled_cfg = _cfg('configs_local/gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py')
    enabled_cfg = _cfg('configs_local/exp_spdi_seed0.py')

    disabled = _build(disabled_cfg).to(device).eval()
    disabled_load = _state(disabled, OFFICIAL_CHECKPOINT, strict=True)
    enabled = _build(enabled_cfg).to(device).eval()
    enabled_load = _state(enabled, OFFICIAL_CHECKPOINT, strict=False)

    torch.manual_seed(2026)
    rgb = torch.randn(1, 3, 512, 640, device=device)
    thermal = torch.randn(1, 3, 512, 640, device=device)
    with torch.no_grad():
        disabled_features = disabled.extract_feat((rgb, thermal))
        enabled_features = enabled.extract_feat((rgb, thermal))
    torch.cuda.synchronize(device)
    zero_init_diff = _max_diff(disabled_features, enabled_features)
    zero_init_equal = all(
        torch.equal(a, b) for a, b in zip(disabled_features, enabled_features))
    detector_level_count = len(disabled_features)

    with torch.no_grad():
        p2_rgb, p2_thermal = enabled.backbone(rgb, thermal)
        detail = enabled.p2_detail_injection.build_detail(
            p2_rgb[0], p2_thermal[0])
    enabled_feature_shapes = [list(feature.shape) for feature in enabled_features]
    shape_result = {
        'backbone_c2_rgb': list(p2_rgb[0].shape),
        'backbone_c2_thermal': list(p2_thermal[0].shape),
        'p2_detail': list(detail.shape),
        'detector_levels': enabled_feature_shapes,
        'detector_level_count': len(enabled_features),
        'expected_detector_level_count': 5,
    }

    # At alpha=0 the branch is identity-preserving. To verify gradients through
    # the branch itself, use a non-zero scale after separately testing zero-init.
    enabled.train()
    enabled.p2_detail_injection.alpha.data.fill_(0.1)
    small_rgb = torch.randn(1, 3, 128, 160, device=device)
    small_thermal = torch.randn(1, 3, 128, 160, device=device)
    enabled.zero_grad(set_to_none=True)
    gradient_features = enabled.extract_feat((small_rgb, small_thermal))
    gradient_loss = sum(feature.square().mean() for feature in gradient_features)
    gradient_loss.backward()
    spdi = enabled.p2_detail_injection
    gradient_result = {
        'alpha_nonzero_finite': _finite_nonzero(spdi.alpha.grad),
        'depthwise_nonzero_finite': _finite_nonzero(spdi.depthwise.weight.grad),
        'pointwise_nonzero_finite': _finite_nonzero(spdi.pointwise.weight.grad),
        'gate_nonzero_finite': _finite_nonzero(spdi.gate.weight.grad),
        'baseline_fusion_nonzero_finite': _finite_nonzero(
            enabled.fuse[0].Q_rgb.conv[0].weight.grad),
        'loss_finite': bool(torch.isfinite(gradient_loss).item()),
        'alpha_at_config_init': 0.0,
        'gradient_test_alpha': 0.1,
    }

    del disabled, enabled, disabled_features, enabled_features
    torch.cuda.empty_cache()
    smoke = _build(enabled_cfg).to(device)
    smoke_load = _state(smoke, OFFICIAL_CHECKPOINT, strict=False)
    smoke = MMDataParallel(smoke, device_ids=[0])
    optimizer = build_optimizer(smoke, enabled_cfg.optimizer)
    dataset = build_dataset(enabled_cfg.data.train)
    smoke.module.CLASSES = dataset.CLASSES
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=8,
        workers_per_gpu=enabled_cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=True)

    smoke_metrics = []
    smoke_status = 'PASS'
    smoke_error = None
    peak_driver = 0
    iterator = iter(data_loader)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for iteration in range(5):
            data = next(iterator)
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            outputs = smoke.train_step(data, optimizer)
            outputs['loss'].backward()
            grad_cfg = enabled_cfg.optimizer_config.get('grad_clip')
            if grad_cfg is not None:
                torch.nn.utils.clip_grad_norm_(
                    smoke.parameters(),
                    max_norm=grad_cfg['max_norm'],
                    norm_type=grad_cfg.get('norm_type', 2))
            optimizer.step()
            torch.cuda.synchronize(device)
            driver = _driver_used_mb()
            peak_driver = max(peak_driver, driver or 0)
            smoke_metrics.append({
                'iteration': iteration + 1,
                'loss': float(outputs['loss'].detach().cpu()),
                'log_vars': {
                    key: float(value)
                    for key, value in outputs['log_vars'].items()},
                'seconds': time.time() - start,
                'driver_used_mb': driver,
            })
            if not torch.isfinite(outputs['loss']).item():
                raise RuntimeError('non-finite loss')
    except RuntimeError as exc:
        smoke_status = 'OOM' if 'out of memory' in str(exc).lower() else 'FAIL'
        smoke_error = repr(exc)

    module = smoke.module.p2_detail_injection
    gradient_checks_pass = all(
        value for key, value in gradient_result.items()
        if isinstance(value, bool))
    prep_pass = (
        zero_init_equal and zero_init_diff == 0.0 and
        gradient_checks_pass and smoke_status == 'PASS' and
        len(smoke_metrics) == 5)
    result = {
        'status': 'PASS' if prep_pass else 'FAIL',
        'configs': {
            'disabled': 'configs_local/gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py',
            'enabled': 'configs_local/exp_spdi_seed0.py',
        },
        'checkpoint': str(OFFICIAL_CHECKPOINT),
        'state_dict_compatibility': {
            'disabled_strict_load': disabled_load,
            'enabled_load': enabled_load,
            'enabled_only_missing_keys': enabled_load['missing_keys'],
            'smoke_load': smoke_load,
        },
        'zero_init_equivalence': {
            'max_abs_diff': zero_init_diff,
            'torch_equal_all_levels': zero_init_equal,
            'detector_level_count': detector_level_count,
        },
        'gradient_test': gradient_result,
        'shapes': shape_result,
        'parameters': _parameter_counts(module),
        'approx_flops': _flops(module),
        'memory_smoke': {
            'status': smoke_status,
            'iterations_requested': 5,
            'iterations_completed': len(smoke_metrics),
            'batch_size': 8,
            'input_resolution': [512, 640],
            'amp': False,
            'gradient_accumulation': False,
            'metrics': smoke_metrics,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'peak_driver_used_mb': peak_driver,
            'error': smoke_error,
        },
        'ready_for_training': prep_pass,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
