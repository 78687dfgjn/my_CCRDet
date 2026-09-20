"""Create a shape-compatible warm-start checkpoint for Phase 3A.

Only parameters with identical names and shapes are copied. Parameters that
are new or whose FPN semantics changed are intentionally reinitialized.
"""
import argparse
import json

import torch
from mmcv import Config
from mmdet.models import build_detector


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

    compatible = {}
    missing = []
    shape_mismatch = []
    for key, value in target.items():
        if key not in source:
            missing.append(key)
        elif tuple(source[key].shape) != tuple(value.shape):
            shape_mismatch.append({
                'key': key,
                'source_shape': list(source[key].shape),
                'target_shape': list(value.shape)
            })
        else:
            compatible[key] = source[key]

    unexpected = sorted(set(source) - set(target))
    newly_initialized = sorted(
        missing + [item['key'] for item in shape_mismatch])
    output_ckpt = {
        'state_dict': compatible,
        'meta': source_ckpt.get('meta', {}),
        'phase3a_warmstart_source': args.source,
    }
    torch.save(output_ckpt, args.output)
    report = {
        'config': args.config,
        'source': args.source,
        'output': args.output,
        'source_keys': len(source),
        'target_keys': len(target),
        'compatible_keys': len(compatible),
        'missing_keys': sorted(missing),
        'shape_mismatch': shape_mismatch,
        'unexpected_keys': unexpected,
        'newly_initialized_keys': newly_initialized,
    }
    with open(args.report, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

