"""Create a semantic P3-P7 to P2-P6 warm-start checkpoint.

The remapper is level-aware. It never falls back to same-name copying for FPN
or fusion keys, because those names change meaning when start_level changes.
"""
import argparse
import json
import re

import torch
from mmcv import Config
from mmdet.models import build_detector


LEVEL_REMAP = {0: 1, 1: 2, 2: 3, 3: 4}
FUSION_REMAP = {0: 1, 1: 2, 2: 3, 3: 4}
FPN_PREFIXES = (
    'neck.lateral_convs.', 'neck.fpn_convs.',
    'nect_t.lateral_convs.', 'nect_t.fpn_convs.')
FUSION_PREFIX = 'fuse.'


def _indexed_key(key, prefixes):
    for prefix in prefixes:
        if key.startswith(prefix):
            match = re.match(r'^' + re.escape(prefix) + r'(\d+)(\..*)$', key)
            if match:
                return prefix, int(match.group(1)), match.group(2)
    return None


def _semantic_source_key(target_key):
    parsed = _indexed_key(target_key, FPN_PREFIXES)
    if parsed is not None:
        prefix, new_index, suffix = parsed
        if 'lateral_convs' in prefix:
            # Actual MMDetection FPN has lateral convolutions only for the
            # backbone inputs (P2-P5 in the candidate). P2 is fresh; P3-P5
            # map to old P3-P5.
            old_index = new_index - 1 if 1 <= new_index <= 3 else None
        else:
            # fpn_convs contains all output levels, including extra P6/P7.
            old_index = new_index - 1 if 1 <= new_index <= 4 else None
        if old_index is None:
            return None, 'p2_or_unmapped_fpn'
        return prefix + str(old_index) + suffix, 'fpn_level_remap'

    parsed = _indexed_key(target_key, (FUSION_PREFIX,))
    if parsed is not None:
        prefix, new_index, suffix = parsed
        old_index = new_index - 1 if 1 <= new_index <= 4 else None
        if old_index is None:
            return None, 'p2_or_unmapped_fusion'
        return prefix + str(old_index) + suffix, 'fusion_level_remap'

    return target_key, 'same_name'


def semantic_warmstart(source, target):
    loaded = {}
    loaded_map = []
    new_initialized = []
    discarded = set()

    for target_key, target_value in target.items():
        source_key, reason = _semantic_source_key(target_key)
        if source_key is None:
            new_initialized.append(target_key)
            continue
        if source_key not in source:
            new_initialized.append(target_key)
            continue
        if tuple(source[source_key].shape) != tuple(target_value.shape):
            raise RuntimeError(
                'semantic mapping shape mismatch: {} <- {}: {} vs {}'.format(
                    target_key, source_key, tuple(target_value.shape),
                    tuple(source[source_key].shape)))
        loaded[target_key] = source[source_key]
        loaded_map.append({
            'target_key': target_key,
            'source_key': source_key,
            'reason': reason,
        })

    used_source = {item['source_key'] for item in loaded_map}
    for source_key in source:
        if source_key not in used_source and (
                source_key.startswith('neck.lateral_convs.') or
                source_key.startswith('neck.fpn_convs.') or
                source_key.startswith('nect_t.lateral_convs.') or
                source_key.startswith('nect_t.fpn_convs.') or
                source_key.startswith('fuse.')):
            discarded.add(source_key)

    return loaded, loaded_map, new_initialized, sorted(discarded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.backbone.pretrained = None
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    source_ckpt = torch.load(args.source, map_location='cpu')
    source = source_ckpt['state_dict']
    target = model.state_dict()

    loaded, loaded_map, new_initialized, discarded = semantic_warmstart(
        source, target)
    full_candidate = dict(target)
    full_candidate.update(loaded)
    candidate_ckpt = {
        'state_dict': full_candidate,
        'meta': source_ckpt.get('meta', {}),
        'phase3a_semantic_warmstart_source': args.source,
        'phase3a_semantic_warmstart_config': args.config,
    }
    torch.save(candidate_ckpt, args.output)

    p2_keys = sorted(
        key for key in new_initialized if (
            key.startswith('neck.lateral_convs.0.') or
            key.startswith('neck.fpn_convs.0.') or
            key.startswith('nect_t.lateral_convs.0.') or
            key.startswith('nect_t.fpn_convs.0.') or
            key.startswith('fuse.0.')))
    p7_discarded = sorted(
        key for key in discarded if (
            key.startswith('neck.fpn_convs.4.') or
            key.startswith('nect_t.fpn_convs.4.') or
            key.startswith('fuse.4.')))
    p2_parameter_count = sum(target[key].numel() for key in p2_keys)
    discarded_parameter_count = sum(
        source[key].numel() for key in discarded if key in source)
    report = {
        'source_checkpoint': args.source,
        'candidate_config': args.config,
        'candidate_checkpoint': args.output,
        'mapping': {
            'P3_old_to_P3_new': True,
            'P4_old_to_P4_new': True,
            'P5_old_to_P5_new': True,
            'P6_old_to_P6_new': True,
            'P7_discard': True,
            'P2_initialized': True,
        },
        'level_index_mapping': {'old_0_to_new_1': True,
                                'old_1_to_new_2': True,
                                'old_2_to_new_3': True,
                                'old_3_to_new_4': True,
                                'old_4_dropped': True},
        'fusion_index_mapping': {'old_0_to_new_1': True,
                                 'old_1_to_new_2': True,
                                 'old_2_to_new_3': True,
                                 'old_3_to_new_4': True,
                                 'old_4_dropped': True},
        'loaded_keys': len(loaded),
        'new_initialized_keys': len(new_initialized),
        'discarded_keys': len(discarded),
        'p2_initialized_parameter_count': p2_parameter_count,
        'discarded_parameter_count': discarded_parameter_count,
        'p2_new_initialized': True,
        'p2_initialized_keys': p2_keys,
        'p7_discarded_keys': p7_discarded,
        'loaded_key_map': loaded_map,
        'new_initialized_key_list': sorted(new_initialized),
        'discarded_key_list': discarded,
    }
    with open(args.report, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
