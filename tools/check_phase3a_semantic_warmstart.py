"""Verify semantic P2-P6 warm-start provenance."""
import argparse
import json

import torch
from mmcv import Config
from mmdet.models import build_detector

from tools.make_phase3a_warmstart import semantic_warmstart


def _equal(a, b):
    return torch.equal(a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--candidate-config', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    source = torch.load(args.source, map_location='cpu')['state_dict']
    candidate = torch.load(args.candidate, map_location='cpu')['state_dict']
    cfg = Config.fromfile(args.candidate_config)
    cfg.model.backbone.pretrained = None
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    target = model.state_dict()
    expected, loaded_map, new_keys, discarded = semantic_warmstart(source, target)
    actual_loaded = []
    failures = []

    for item in loaded_map:
        target_key = item['target_key']
        source_key = item['source_key']
        if target_key not in candidate:
            failures.append('candidate missing loaded key: ' + target_key)
            continue
        if not _equal(candidate[target_key], source[source_key]):
            failures.append('{} does not equal {}'.format(target_key, source_key))
        actual_loaded.append(target_key)

    expected_new = set(new_keys)
    p2_prefixes = (
        'neck.lateral_convs.0.', 'neck.fpn_convs.0.',
        'nect_t.lateral_convs.0.', 'nect_t.fpn_convs.0.', 'fuse.0.')
    p2_keys = sorted(k for k in expected_new if k.startswith(p2_prefixes))
    p2_pass = bool(p2_keys) and all(k not in actual_loaded for k in p2_keys)
    if not p2_pass:
        failures.append('P2 keys were not all excluded from source loading')

    p7_pairs = [
        ('neck.fpn_convs.4.conv.weight', 'neck.fpn_convs.3.conv.weight'),
        ('neck.fpn_convs.4.conv.bias', 'neck.fpn_convs.3.conv.bias'),
        ('nect_t.fpn_convs.4.conv.weight', 'nect_t.fpn_convs.3.conv.weight'),
        ('nect_t.fpn_convs.4.conv.bias', 'nect_t.fpn_convs.3.conv.bias'),
        ('fuse.4.conv1x1.weight', 'fuse.3.conv1x1.weight'),
        ('fuse.4.conv1x1.bias', 'fuse.3.conv1x1.bias'),
    ]
    p7_pass = True
    for target_key, p6_source_key in p7_pairs:
        if target_key not in candidate or not _equal(candidate[target_key], source[p6_source_key]):
            p7_pass = False
            failures.append('{} is not remapped from {}'.format(target_key, p6_source_key))
        if target_key in source and _equal(candidate[target_key], source[target_key]):
            p7_pass = False
            failures.append('{} still equals discarded old P7 tensor'.format(target_key))

    config_fusion = list(cfg.model.fusion_types)
    fusion_pass = config_fusion == [
        'fusion_cat', 'fusion', 'fusion', 'fusion', 'fusion_cat']
    if not fusion_pass:
        failures.append('candidate fusion_types are not P2-P6 semantic types')

    semantic_failures = list(failures)

    result = {
        'status': 'PASS' if not failures else 'FAIL',
        'source_checkpoint': args.source,
        'candidate_checkpoint': args.candidate,
        'loaded_key_count': len(actual_loaded),
        'expected_loaded_key_count': len(expected),
        'new_initialized_key_count': len(new_keys),
        'p2_initialized_key_count': len(p2_keys),
        'p2_initialized_parameter_count': sum(
            target[key].numel() for key in p2_keys),
        'discarded_key_count': len(discarded),
        'checks': {
            'p3_to_p6_semantic_values': not semantic_failures,
            'p2_not_source_loaded': p2_pass,
            'p7_discarded_and_p6_target_remapped': p7_pass,
            'candidate_fusion_types': fusion_pass,
        },
        'p2_initialized_keys': p2_keys,
        'discarded_keys': discarded,
        'failures': failures,
    }
    with open(args.report, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
