"""Zero-logit PGCF telemetry aggregation audit."""
import json
from pathlib import Path

import torch

from mmdet.models.utils.p2_groupwise_complementary_fusion import (
    P2GroupwiseComplementaryFusion)


def main():
    torch.manual_seed(8)
    module = P2GroupwiseComplementaryFusion(
        256, groups=16, analysis_enabled=True, checkpoint_enabled=False)
    module.eval()
    rgb = torch.randn(2, 256, 4, 5)
    thermal = torch.randn(2, 256, 4, 5)
    with torch.no_grad():
        module(rgb, thermal)
    stats = module.get_analysis_stats()
    group_errors = [
        abs(value - 1.0) for value in stats['group_mean_rgb_weight'] +
        stats['group_mean_thermal_weight']]
    result = {
        'status': 'PASS' if (
            abs(stats['mean_rgb_weight'] - 1.0) <= 1e-6 and
            abs(stats['mean_thermal_weight'] - 1.0) <= 1e-6 and
            max(group_errors) <= 1e-6
        ) else 'FAIL',
        'root_cause': (
            'Previous smoke report divided group sums by the total count '
            'across all groups; the group aggregation denominator was '
            'count/groups. Forward math was unchanged.'),
        'mean_rgb_weight': stats['mean_rgb_weight'],
        'mean_thermal_weight': stats['mean_thermal_weight'],
        'group_mean_rgb_weight': stats['group_mean_rgb_weight'],
        'group_mean_thermal_weight': stats['group_mean_thermal_weight'],
        'max_group_abs_error': max(group_errors),
        'telemetry_calls': stats['calls'],
    }
    Path('experiments').mkdir(exist_ok=True)
    Path('experiments/phase3h_telemetry_audit.json').write_text(
        json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
