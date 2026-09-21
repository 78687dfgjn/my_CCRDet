"""Audit Phase 3H protocol and PGCF warm-start provenance."""
import hashlib
import json
from pathlib import Path

import torch
from mmcv import Config
from mmdet.models import build_detector


SOURCE = '/hy-tmp/CCRDet_assets/phase3a_semantic_warmstart_seed0.pth'
CONTROL_CONFIG = 'configs_local/exp_proxy_lite_p2p6_seed0.py'
CANDIDATE_CONFIG = 'configs_local/exp_proxy_pgcf_p2p6_seed0.py'
ALLOWED_MISSING = (
    'fuse.0.reduce.',
    'fuse.0.depthwise.',
    'fuse.0.gate_logits.',
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def value(cfg, key):
    item = cfg
    for part in key.split('.'):
        if isinstance(item, dict):
            if part not in item:
                return '<MISSING>'
            item = item[part]
        else:
            try:
                item = getattr(item, part)
            except AttributeError:
                return '<MISSING>'
    return repr(item)



def build_candidate():
    cfg = Config.fromfile(CANDIDATE_CONFIG)
    cfg.model.backbone.pretrained = None
    cfg.model.pop('pretrained', None)
    cfg.model.pop('init_cfg', None)
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    source = torch.load(SOURCE, map_location='cpu')['state_dict']
    incompat = model.load_state_dict(source, strict=False)
    return cfg, model, source, list(incompat.missing_keys), list(
        incompat.unexpected_keys)


def main():
    source_hash = sha256(SOURCE)
    control = Config.fromfile(CONTROL_CONFIG)
    cfg, model, source, missing, unexpected = build_candidate()
    target = model.state_dict()
    extra_keys = sorted(set(target) - set(source))
    source_not_target = sorted(set(source) - set(target))
    common_failures = []
    for key in sorted(set(source) & set(target)):
        if not torch.equal(source[key], target[key]):
            common_failures.append(key)

    protocol_fields = [
        'data', 'optimizer', 'optimizer_config', 'lr_config', 'runner',
        'evaluation', 'seed', 'deterministic', 'train_cfg', 'test_cfg']
    protocol_equal = {
        field: value(control, field) == value(cfg, field)
        for field in protocol_fields
    }
    architecture = {
        'control_fusion_types': list(control.model.fusion_types),
        'candidate_fusion_types': list(cfg.model.fusion_types),
        'control_fpn_start_level': int(control.model.neck.start_level),
        'candidate_fpn_start_level': int(cfg.model.neck.start_level),
        'control_num_outs': int(control.model.neck.num_outs),
        'candidate_num_outs': int(cfg.model.neck.num_outs),
        'control_strides': list(
            control.model.bbox_head.anchor_generator.strides),
        'candidate_strides': list(
            cfg.model.bbox_head.anchor_generator.strides),
        'candidate_pgcf_analysis': bool(cfg.model.pgcf_analysis),
        'candidate_pgcf_checkpoint': bool(cfg.model.pgcf_checkpoint),
        'global_shift': repr(cfg.model.global_shift),
        'p2_detail': repr(cfg.model.p2_detail),
        'p2_alignment': repr(cfg.model.p2_alignment),
    }

    final_conv = model.fuse[0].conv1x1
    existing_params = sum(parameter.numel() for parameter in final_conv.parameters())
    pgcf_params = sum(parameter.numel() for parameter in model.fuse[0].parameters())
    height, width = 128, 160
    pixels = height * width
    reduce_mac = pixels * 768 * 32
    depthwise_mac = pixels * 32 * 9
    gate_mac = pixels * 32 * 32
    additional_mac = reduce_mac + depthwise_mac + gate_mac
    result = {
        'status': 'PASS' if (
            all(protocol_equal.values()) and not unexpected and
            not source_not_target and not common_failures and
            all(any(key.startswith(prefix) for prefix in ALLOWED_MISSING)
                for key in missing)
        ) else 'FAIL',
        'source_checkpoint': SOURCE,
        'source_sha256': source_hash,
        'control_config': CONTROL_CONFIG,
        'candidate_config': CANDIDATE_CONFIG,
        'warmstart': {
            'missing_keys': missing,
            'unexpected_keys': unexpected,
            'allowed_new_key_prefixes': list(ALLOWED_MISSING),
            'source_not_in_candidate': source_not_target,
            'common_tensor_mismatches': common_failures,
            'source_common_tensor_count': len(set(source) & set(target)),
            'pgcf_fresh_keys': extra_keys,
            'pgcf_fresh_key_count': len(extra_keys),
        },
        'protocol_equal_to_control': protocol_equal,
        'architecture': architecture,
        'parameters': {
            'fusion_cat_params': int(existing_params),
            'pgcf_total_params': int(pgcf_params),
            'pgcf_additional_params': int(pgcf_params - existing_params),
            'reduce_params': int(sum(
                parameter.numel()
                for parameter in model.fuse[0].reduce.parameters())),
            'depthwise_params': int(sum(
                parameter.numel()
                for parameter in model.fuse[0].depthwise.parameters())),
            'gate_logits_params': int(sum(
                parameter.numel()
                for parameter in model.fuse[0].gate_logits.parameters())),
        },
        'flops': {
            'definition': 'multiply-add counted as 2 FLOPs; conv-only',
            'p2_shape': [height, width],
            'fusion_cat_conv_flops': int(2 * pixels * 512 * 256),
            'pgcf_additional_conv_flops': int(2 * additional_mac),
            'pgcf_total_conv_flops': int(
                2 * (pixels * 512 * 256 + additional_mac)),
            'excluded': 'abs, softmax, ReLU, tensor reshaping',
        },
    }
    Path('experiments').mkdir(exist_ok=True)
    Path('experiments/phase3h_protocol_audit.json').write_text(
        json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
