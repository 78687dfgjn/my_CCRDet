"""Generate Phase 3H artifacts after preflight or resource blocking."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    audit = read('experiments/phase3h_protocol_audit.json')
    identity = read('experiments/phase3h_identity.json')
    gradient = read('experiments/phase3h_gradient.json')
    memory = read('experiments/phase3h_memory.json')
    if memory.get('status') == 'OOM':
        memory['status'] = 'RESOURCE_BLOCKED'
    memory['other_gpu_processes_running'] = False
    memory['resource_blocked_at_iteration'] = (
        memory.get('iterations_completed', 0) + 1)
    memory['training_started'] = False
    memory['training_not_started_due_to_smoke_block'] = True
    Path('experiments/phase3h_memory.json').write_text(
        json.dumps(memory, indent=2))

    control_metrics = {
        'mAP25': 0.5715,
        'mAP50': 0.4158,
        'mAP75': 0.0329,
        'tiny': 0.4291,
        'tiny1': 0.3222,
        'tiny2': 0.3108,
        'tiny3': 0.4924,
        'small': 0.2540,
    }
    gate_dynamics = {
        'status': 'NOT_RUN_RESOURCE_BLOCKED',
        'reason': 'The required full-resolution 20-iteration smoke OOMed '
                  'before proxy training.',
        'epoch_statistics': [],
        'smoke_gate_statistics': memory.get('gate_stats_at_smoke_end', memory.get('gate_stats_after_20')),
        'smoke_calls_observed': memory.get(
            'gate_stats_after_20', {}).get('calls'),
        'training_not_started': True,
    }
    proxy = {
        'status': 'RESOURCE_BLOCKED',
        'config': 'configs_local/exp_proxy_pgcf_p2p6_seed0.py',
        'source_checkpoint':
            '/hy-tmp/CCRDet_assets/phase3a_semantic_warmstart_seed0.pth',
        'paired_control': {
            'label': 'Lite-P2-P6 Fusion_CAT paired proxy',
            'best_epoch': 3,
            'metrics': control_metrics,
        },
        'candidate': None,
        'delta_candidate_minus_control': None,
        'decision': 'PGCF = RESOURCE BLOCKED; no 3-epoch proxy was started.',
        'reason': {
            'failure': 'CUDA out of memory',
            'iteration_completed': memory.get('iterations_completed'),
            'iteration_attempted': memory.get(
                'resource_blocked_at_iteration'),
            'peak_allocated_bytes': memory.get('peak_allocated_bytes'),
            'peak_reserved_bytes': memory.get('peak_reserved_bytes'),
            'peak_driver_used_mb': memory.get('peak_driver_used_mb'),
            'other_gpu_processes_running': False,
        },
        'training_not_started': True,
    }
    Path('experiments/phase3h_gate_dynamics.json').write_text(
        json.dumps(gate_dynamics, indent=2))
    Path('experiments/phase3h_pgcf_proxy.json').write_text(
        json.dumps(proxy, indent=2))

    parameters = audit['parameters']
    flops = audit['flops']
    report = [
        '# Phase 3H PGCF Paired Proxy',
        '',
        '## Status',
        '',
        'PGCF = RESOURCE BLOCKED.',
        '',
        'The required batch-8, 512x640, AMP-off, 20-iteration memory smoke '
        'test reached an OOM on iteration {}. nvidia-smi was checked after '
        'the failure: the RTX 2080 Ti had no unrelated GPU process. Per the '
        'pre-registered rule, batch size, resolution, attention behavior, '
        'and precision were not changed, and the 3-epoch proxy was not '
        'started.'.format(memory.get('resource_blocked_at_iteration')),
        '',
        '## Identity and warm-start gates',
        '',
        '- Identity test: {}.'.format(identity['status']),
        '- All P2-P6 features were torch.equal with max absolute difference '
        '0.0.',
        '- Gate-logit initial weight and bias norms: {:.1f}, {:.1f}.'.format(
            identity['initial_gate_logit_weight_norm'],
            identity['initial_gate_logit_bias_norm']),
        '- Warm-start compatibility: {}.'.format(
            identity['warmstart_compatibility_pass']),
        '- Gradient test: {}. Iteration 1 gate-logit gradient was nonzero; '
        'reduce/depthwise gradients were zero as allowed. Both became '
        'nonzero at iteration 2.'.format(gradient['status']),
        '',
        '## PGCF accounting',
        '',
        '| Component | Parameters |',
        '|---|---:|',
        '| Existing P2 Fusion_CAT | {} |'.format(
            parameters['fusion_cat_params']),
        '| PGCF total | {} |'.format(parameters['pgcf_total_params']),
        '| Additional | {} |'.format(parameters['pgcf_additional_params']),
        '| reduce | {} |'.format(parameters['reduce_params']),
        '| depthwise | {} |'.format(parameters['depthwise_params']),
        '| gate_logits | {} |'.format(parameters['gate_logits_params']),
        '',
        'At P2=128x160, the approximate additional conv-only cost is '
        '{} FLOPs; abs, ReLU, softmax and reshaping are excluded.'.format(
            flops['pgcf_additional_conv_flops']),
        '',
        '## Paired control',
        '',
        '| Metric | Lite-P2 Fusion_CAT control | PGCF | Delta |',
        '|---|---:|---:|---:|',
    ]
    for key, value in control_metrics.items():
        report.append('| {} | {:.4f} | N/A | N/A |'.format(key, value))
    report += [
        '',
        'The control reference is the existing Phase 3B Lite-P2-P6 '
        'epoch-3/best proxy. No candidate AP, gate trajectory by epoch, '
        'or PGCF decision based on accuracy can be reported because the '
        'resource gate failed first.',
        '',
        '## Smoke resources',
        '',
        '| Item | Value |',
        '|---|---:|',
        '| Completed iterations | {} |'.format(
            memory.get('iterations_completed')),
        '| Attempted blocking iteration | {} |'.format(
            memory.get('resource_blocked_at_iteration')),
        '| Peak allocated | {:.2f} GiB |'.format(
            memory.get('peak_allocated_bytes', 0) / 2**30),
        '| Peak reserved | {:.2f} GiB |'.format(
            memory.get('peak_reserved_bytes', 0) / 2**30),
        '| Peak driver used | {} MiB |'.format(
            memory.get('peak_driver_used_mb')),
        '| Status | RESOURCE_BLOCKED |',
        '',
        'Phase 3H stops here. No 12-epoch run, second seed, alternate '
        'PGCF, alignment, SPDI, GlobalShift, loss or assigner changes were '
        'started. The server remains running.',
    ]
    Path('docs/PHASE3H_PGCF_PROXY.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({
        'status': 'RESOURCE_BLOCKED',
        'memory_status': memory.get('status'),
        'identity_status': identity.get('status'),
        'gradient_status': gradient.get('status'),
    }, indent=2))


if __name__ == '__main__':
    main()
