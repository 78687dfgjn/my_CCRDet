"""Checkpointed/non-checkpointed PGCF equivalence test."""
import json
from pathlib import Path

import torch

from mmdet.models.utils.p2_groupwise_complementary_fusion import (
    P2GroupwiseComplementaryFusion)


def main():
    torch.manual_seed(7)
    module_a = P2GroupwiseComplementaryFusion(
        256, groups=16, analysis_enabled=False, checkpoint_enabled=False)
    module_b = P2GroupwiseComplementaryFusion(
        256, groups=16, analysis_enabled=False, checkpoint_enabled=True)
    module_b.load_state_dict(module_a.state_dict())
    module_a.train()
    module_b.train()

    rgb_data = torch.randn(2, 256, 16, 20)
    thermal_data = torch.randn(2, 256, 16, 20)
    rgb_a = rgb_data.clone().requires_grad_(True)
    thermal_a = thermal_data.clone().requires_grad_(True)
    rgb_b = rgb_data.clone().requires_grad_(True)
    thermal_b = thermal_data.clone().requires_grad_(True)

    output_a = module_a(rgb_a, thermal_a)
    output_b = module_b(rgb_b, thermal_b)
    output_diff = float((output_a - output_b).abs().max().item())
    output_equal = bool(torch.equal(output_a, output_b))

    loss_a = (output_a * output_a).mean()
    loss_b = (output_b * output_b).mean()
    loss_a.backward()
    loss_b.backward()

    gradient_rows = []
    max_gradient_diff = 0.0
    for name in ('reduce.weight', 'depthwise.weight',
                 'gate_logits.weight', 'conv1x1.weight'):
        left = dict(module_a.named_parameters())[name].grad
        right = dict(module_b.named_parameters())[name].grad
        diff = float((left - right).abs().max().item())
        max_gradient_diff = max(max_gradient_diff, diff)
        gradient_rows.append({
            'parameter': name,
            'finite_a': bool(torch.isfinite(left).all().item()),
            'finite_b': bool(torch.isfinite(right).all().item()),
            'max_abs_gradient_diff': diff,
            'gradient_norm_a': float(left.norm().item()),
            'gradient_norm_b': float(right.norm().item()),
        })

    result = {
        'status': 'PASS' if (
            output_diff <= 1e-6 and
            all(row['finite_a'] and row['finite_b'] and
                row['max_abs_gradient_diff'] <= 1e-6
                for row in gradient_rows)
        ) else 'FAIL',
        'input_shape': [2, 256, 16, 20],
        'checkpoint_enabled_a': False,
        'checkpoint_enabled_b': True,
        'output_torch_equal': output_equal,
        'output_max_abs_diff': output_diff,
        'gradient_rows': gradient_rows,
        'max_abs_gradient_diff': max_gradient_diff,
        'telemetry_calls_a': module_a.get_analysis_stats()['calls'],
        'telemetry_calls_b': module_b.get_analysis_stats()['calls'],
        'telemetry_recomputation_not_counted': (
            module_b.get_analysis_stats()['calls'] == 0),
    }
    Path('experiments').mkdir(exist_ok=True)
    Path('experiments/phase3h_checkpoint_equivalence.json').write_text(
        json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
