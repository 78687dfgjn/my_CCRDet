"""Verify formal-resolution Phase 3B candidate fusion paths."""
import json
from pathlib import Path

import torch
from mmcv import Config
from mmdet.models import build_detector
from mmdet.models.detectors.afdet import Fusion, Fusion_CAT


def main():
    config_path = 'configs_local/exp_p2p6_seed0.py'
    report_path = Path('experiments/phase3b_fusion_path_check.json')
    cfg = Config.fromfile(config_path)
    cfg.model.backbone.pretrained = None
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    expected_types = ['Fusion_CAT', 'Fusion', 'Fusion', 'Fusion', 'Fusion_CAT']
    expected_shapes = [
        [1, 256, 128, 160], [1, 256, 64, 80],
        [1, 256, 32, 40], [1, 256, 16, 20], [1, 256, 8, 10]]

    actual_types = [type(module).__name__ for module in model.fuse]
    for module in model.fuse:
        if isinstance(module, Fusion):
            module.analysis_enabled = True

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rgb = torch.randn(1, 3, 512, 640, device=device)
    thermal = torch.randn(1, 3, 512, 640, device=device)
    with torch.no_grad():
        features = model.extract_feat((rgb, thermal))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    shapes = [list(feature.shape) for feature in features]
    fusion_stats = []
    for index, module in enumerate(model.fuse):
        stats = {
            'index': index,
            'semantic_level': 'P{}'.format(index + 2),
            'module_type': type(module).__name__,
            'spatial_shape': shapes[index][-2:],
        }
        if isinstance(module, Fusion):
            stats.update({
                'attention_mode': module.attention_mode,
                'threshold': module.attention_hw_threshold,
                'analysis_total_calls': module.analysis_total_calls,
                'analysis_full_calls': module.analysis_full_calls,
                'analysis_skip_calls': module.analysis_skip_calls,
            })
        fusion_stats.append(stats)

    fusion_modules_pass = actual_types == expected_types
    shapes_pass = shapes == expected_shapes
    p3 = fusion_stats[1]
    p3_full_pass = (
        p3['module_type'] == 'Fusion' and
        p3['attention_mode'] == 'full' and
        p3['analysis_full_calls'] == 1 and
        p3['analysis_skip_calls'] == 0)
    result = {
        'status': 'PASS' if fusion_modules_pass and shapes_pass and p3_full_pass
        else 'FAIL',
        'config': config_path,
        'input_shapes': {
            'rgb': [1, 3, 512, 640],
            'thermal': [1, 3, 512, 640],
        },
        'expected_module_types': expected_types,
        'actual_module_types': actual_types,
        'expected_feature_shapes': expected_shapes,
        'actual_feature_shapes': shapes,
        'fusion_levels': fusion_stats,
        'p3_did_not_skip': p3_full_pass,
        'peak_allocated_bytes': (
            torch.cuda.max_memory_allocated(device) if device.type == 'cuda'
            else None),
        'peak_reserved_bytes': (
            torch.cuda.max_memory_reserved(device) if device.type == 'cuda'
            else None),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
