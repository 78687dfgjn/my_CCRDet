"""Validate Phase 3G semantic warm-start checkpoints."""
import argparse
import json
from pathlib import Path

import torch


def state(path):
    ckpt = torch.load(path, map_location='cpu')
    return ckpt.get('state_dict', ckpt)


def check_group(src, dst, prefix, source_index, target_index, checks):
    source_prefix = prefix + str(source_index) + '.'
    target_prefix = prefix + str(target_index) + '.'
    keys = sorted(k for k in src if k.startswith(source_prefix))
    ok = bool(keys)
    for source_key in keys:
        target_key = target_prefix + source_key[len(source_prefix):]
        equal = target_key in dst and torch.equal(
            src[source_key], dst[target_key])
        checks.append({
            'source_key': source_key,
            'target_key': target_key,
            'torch_equal': bool(equal),
        })
        ok = ok and equal
    return ok


def validate(mode, source_path, candidate_path, report_path):
    src, dst = state(source_path), state(candidate_path)
    checks = []
    passed = True
    if mode == 'p3p6':
        for prefix, count in (
                ('neck.lateral_convs.', 3),
                ('nect_t.lateral_convs.', 3),
                ('neck.fpn_convs.', 4),
                ('nect_t.fpn_convs.', 4),
                ('fuse.', 4)):
            for i in range(count):
                passed &= check_group(src, dst, prefix, i, i, checks)
        discarded = [
            k for k in src
            if k.startswith(('fuse.4.', 'neck.fpn_convs.4.',
                             'nect_t.fpn_convs.4.'))]
        checks.append({'P7_source_keys_discarded': bool(discarded)})
        passed &= bool(discarded)
        passed &= not any(
            k.startswith(('fuse.4.', 'neck.fpn_convs.4.',
                          'nect_t.fpn_convs.4.')) for k in dst)
    elif mode == 'p2p7':
        for prefix, count in (
                ('neck.lateral_convs.', 3),
                ('nect_t.lateral_convs.', 3),
                ('neck.fpn_convs.', 5),
                ('nect_t.fpn_convs.', 5),
                ('fuse.', 5)):
            for i in range(count):
                passed &= check_group(src, dst, prefix, i, i + 1, checks)
        for prefix in ('neck.lateral_convs.', 'nect_t.lateral_convs.',
                       'neck.fpn_convs.', 'nect_t.fpn_convs.', 'fuse.'):
            fresh_keys = [k for k in dst if k.startswith(prefix + '0.')]
            checks.append({
                'fresh_target_level0_present': bool(fresh_keys),
                'prefix': prefix,
            })
            passed &= bool(fresh_keys)
        old_p7 = [
            k for k in src
            if k.startswith(('neck.fpn_convs.4.', 'nect_t.fpn_convs.4.',
                             'fuse.4.'))]
        checks.append({'P7_source_keys_retained': bool(old_p7)})
        passed &= bool(old_p7)
    else:
        raise ValueError(mode)
    result = {
        'status': 'PASS' if passed else 'FAIL',
        'mode': mode,
        'source_checkpoint': str(Path(source_path).resolve()),
        'candidate_checkpoint': str(Path(candidate_path).resolve()),
        'checks': checks,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'status': result['status'],
        'mode': mode,
        'checks': len(checks),
    }, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['p3p6', 'p2p7'], required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    validate(args.mode, args.source, args.candidate, args.report)
