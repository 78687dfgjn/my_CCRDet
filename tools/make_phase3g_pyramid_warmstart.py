"""Build Phase 3G semantic warm-start checkpoints.

This tool is deliberately separate from the Phase 3A P2-P6 remapper.  It
never falls back to same-index copying for indexed FPN/fusion modules: every
indexed tensor is mapped by the declared semantic level map.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from mmcv import Config
from mmdet.models import build_detector


INDEXED_PREFIXES = (
    'neck.lateral_convs.',
    'neck.fpn_convs.',
    'nect_t.lateral_convs.',
    'nect_t.fpn_convs.',
    'fuse.',
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split_indexed_key(key):
    for prefix in INDEXED_PREFIXES:
        if key.startswith(prefix):
            tail = key[len(prefix):]
            token = tail.split('.', 1)[0]
            if token.isdigit():
                return prefix, int(token), tail[len(token):]
    return None


def source_key_for(target_key, mode):
    parsed = split_indexed_key(target_key)
    if parsed is None:
        return target_key
    prefix, target_index, suffix = parsed
    if mode == 'p3p6':
        if target_index < 0 or target_index > 3:
            return None
        source_index = target_index
    elif mode == 'p2p7':
        if prefix.endswith('lateral_convs.'):
            if target_index == 0:
                return None
            if 1 <= target_index <= 3:
                source_index = target_index - 1
            else:
                return None
        else:
            if target_index == 0:
                return None
            if 1 <= target_index <= 5:
                source_index = target_index - 1
            else:
                return None
    else:
        raise ValueError('mode must be p3p6 or p2p7')
    return prefix + str(source_index) + suffix


def build_report(cfg_path, source_path, output_path, report_path, mode):
    cfg = Config.fromfile(cfg_path)
    cfg.model.backbone.pretrained = None
    cfg.model.pop('pretrained', None)
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    target_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }

    source_ckpt = torch.load(source_path, map_location='cpu')
    source_state = source_ckpt.get('state_dict', source_ckpt)
    source_state = {
        k: v.detach().cpu() if torch.is_tensor(v) else v
        for k, v in source_state.items()
    }

    loaded = []
    initialized = []
    mapping_records = []
    used_source = set()
    for target_key in sorted(target_state):
        source_key = source_key_for(target_key, mode)
        record = {
            'target_key': target_key,
            'source_key': source_key,
            'status': 'fresh_initialized',
        }
        if (source_key is not None and source_key in source_state and
                torch.is_tensor(source_state[source_key]) and
                tuple(source_state[source_key].shape) ==
                tuple(target_state[target_key].shape)):
            target_state[target_key] = source_state[source_key].clone()
            used_source.add(source_key)
            loaded.append(target_key)
            record['status'] = 'loaded'
        else:
            initialized.append(target_key)
        mapping_records.append(record)

    missing, unexpected = model.load_state_dict(target_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            'candidate load failed: missing={}, unexpected={}'.format(
                missing, unexpected))

    discarded = sorted(set(source_state) - used_source)
    semantic_checks = {}
    if mode == 'p3p6':
        semantic_checks = {
            'P3_P6_all_equal': all(
                any(r['target_key'].startswith(prefix + str(i) + '.')
                    and r['source_key'].startswith(prefix + str(i) + '.')
                    and r['status'] == 'loaded'
                    for r in mapping_records)
                for prefix in (
                    'neck.fpn_convs.', 'nect_t.fpn_convs.', 'fuse.')
                for i in range(4)),
            'P7_source_fpn_discarded': any(
                k.startswith('neck.fpn_convs.4.') for k in discarded),
            'P7_source_thermal_fpn_discarded': any(
                k.startswith('nect_t.fpn_convs.4.') for k in discarded),
            'P7_source_fuse_discarded': any(
                k.startswith('fuse.4.') for k in discarded),
        }
    else:
        semantic_checks = {
            'P3_P7_all_equal': all(
                any(r['target_key'].startswith(prefix + str(i + 1) + '.')
                    and r['source_key'].startswith(prefix + str(i) + '.')
                    and r['status'] == 'loaded'
                    for r in mapping_records)
                for prefix in (
                    'neck.fpn_convs.', 'nect_t.fpn_convs.', 'fuse.')
                for i in range(5)),
            'P2_lateral_fresh': all(
                r['status'] == 'fresh_initialized'
                for r in mapping_records
                if r['target_key'].startswith('neck.lateral_convs.0.')),
            'P2_thermal_lateral_fresh': all(
                r['status'] == 'fresh_initialized'
                for r in mapping_records
                if r['target_key'].startswith('nect_t.lateral_convs.0.')),
            'P2_fpn_fresh': all(
                r['status'] == 'fresh_initialized'
                for r in mapping_records
                if r['target_key'].startswith('neck.fpn_convs.0.')),
            'P2_thermal_fpn_fresh': all(
                r['status'] == 'fresh_initialized'
                for r in mapping_records
                if r['target_key'].startswith('nect_t.fpn_convs.0.')),
            'P2_fuse_fresh': all(
                r['status'] == 'fresh_initialized'
                for r in mapping_records
                if r['target_key'].startswith('fuse.0.')),
            'P7_retained': all(
                any(r['target_key'].startswith(prefix + '5.')
                    and r['source_key'].startswith(prefix + '4.')
                    and r['status'] == 'loaded'
                    for r in mapping_records)
                for prefix in (
                    'neck.fpn_convs.', 'nect_t.fpn_convs.', 'fuse.')),
            'P7_not_discarded': not any(
                k.startswith(('neck.fpn_convs.4.', 'nect_t.fpn_convs.4.',
                              'fuse.4.')) for k in discarded),
        }

    report = {
        'status': 'PASS' if all(semantic_checks.values()) else 'FAIL',
        'mode': mode,
        'source_checkpoint': str(Path(source_path).resolve()),
        'source_sha256': sha256_file(source_path),
        'target_config': str(Path(cfg_path).resolve()),
        'output_checkpoint': str(Path(output_path).resolve()),
        'mapping': semantic_checks,
        'loaded_keys': len(loaded),
        'new_initialized_keys': len(initialized),
        'discarded_keys': len(discarded),
        'loaded_key_names': loaded,
        'new_initialized_key_names': initialized,
        'discarded_key_names': discarded,
        'mapping_records': mapping_records,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'meta': copy.deepcopy(source_ckpt.get('meta', {})),
        'state_dict': target_state,
        'phase3g_semantic_warmstart': report,
    }, output_path)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    print(json.dumps({
        'status': report['status'],
        'mode': mode,
        'loaded_keys': len(loaded),
        'new_initialized_keys': len(initialized),
        'discarded_keys': len(discarded),
        'output_checkpoint': output_path,
    }, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['p3p6', 'p2p7'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    build_report(
        args.config, args.source, args.output, args.report, args.mode)


if __name__ == '__main__':
    main()
