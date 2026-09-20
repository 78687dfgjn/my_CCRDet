"""Phase 3C formal Lite-P2 initialization audit and five-step preflight.

The preflight is a resource/sanity check only. It starts from the configured
two-stream pretrained backbone and never loads a baseline or warm-start model.
"""

import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from mmcv import Config
import mmcv.utils.config as mmcv_config
from mmcv.parallel import MMDataParallel
from mmdet.apis import set_random_seed
from mmdet.core import build_optimizer
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


# MMCV 1.6.1 calls yapf with verify=, while the installed yapf removed it.
# This affects only config serialization in this audit script.
_format_code = mmcv_config.FormatCode


def _format_code_compat(text, style_config=None, verify=None):
    return _format_code(text, style_config=style_config)


mmcv_config.FormatCode = _format_code_compat


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs_local' / 'exp_full_lite_p2p6_seed0.py'
BASELINE_CONFIG_PATH = (
    ROOT / 'configs_local' / 'gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py')
PRETRAIN_RELATIVE = Path('pretrain_weights/resnet50-2stream.pth')
AUDIT_PATH = ROOT / 'experiments' / 'phase3c_initialization_audit.json'
PREFLIGHT_PATH = ROOT / 'experiments' / 'phase3c_preflight.json'
DIFF_PATH = ROOT / 'experiments' / 'phase3c_resolved_config_diff.txt'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _driver_used_mb():
    try:
        text = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.STDOUT, text=True)
        return int(text.strip().splitlines()[0])
    except Exception:
        return None


def _gpu_processes():
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'],
            stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return 'unavailable: {}'.format(exc)


def _build(cfg):
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    return model


def _effective_fusion_types(cfg):
    return list(cfg.model.get(
        'fusion_types',
        ['fusion', 'fusion', 'fusion', 'fusion_cat', 'fusion_cat']))


def _effective_attention_mode(model):
    return [
        (type(module).__name__, getattr(module, 'attention_mode', None))
        for module in model.fuse]


def _jsonable(value):
    if isinstance(value, Config):
        return value.to_dict()
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    return value


def _protocol_diff(baseline, candidate):
    lines = [
        'Phase 3C resolved-config comparison',
        'baseline = {}'.format(BASELINE_CONFIG_PATH),
        'candidate = {}'.format(CONFIG_PATH),
        '',
        '[unchanged training protocol checks]',
    ]
    checks = [
        ('optimizer', baseline.optimizer, candidate.optimizer),
        ('optimizer_config', baseline.optimizer_config,
         candidate.optimizer_config),
        ('lr_config', baseline.lr_config, candidate.lr_config),
        ('runner', baseline.runner, candidate.runner),
        ('train_pipeline', baseline.data.train.pipeline,
         candidate.data.train.pipeline),
        ('val_pipeline', baseline.data.val.pipeline,
         candidate.data.val.pipeline),
        ('train_cfg', baseline.model.train_cfg, candidate.model.train_cfg),
        ('test_cfg', baseline.model.test_cfg, candidate.model.test_cfg),
        ('data.samples_per_gpu', baseline.data.samples_per_gpu,
         candidate.data.samples_per_gpu),
        ('data.workers_per_gpu', baseline.data.workers_per_gpu,
         candidate.data.workers_per_gpu),
        ('backbone', baseline.model.backbone, candidate.model.backbone),
    ]
    protocol_diffs = []
    for name, left, right in checks:
        equal = _jsonable(left) == _jsonable(right)
        lines.append('{}: {}'.format(name, 'IDENTICAL' if equal else 'DIFF'))
        if not equal:
            protocol_diffs.append(name)

    lines.extend([
        '',
        '[allowed structural differences]',
        'neck.start_level: {} -> {}'.format(
            baseline.model.neck.start_level,
            candidate.model.neck.start_level),
        'neck.num_outs: {} -> {}'.format(
            baseline.model.neck.num_outs, candidate.model.neck.num_outs),
        'bbox_head.anchor_generator.strides: {} -> {}'.format(
            list(baseline.model.bbox_head.anchor_generator.strides),
            list(candidate.model.bbox_head.anchor_generator.strides)),
        'fusion_types: {} -> {}'.format(
            _effective_fusion_types(baseline),
            _effective_fusion_types(candidate)),
        'global_shift: {} -> {}'.format(
            baseline.model.get('global_shift', {'mode': 'off'}),
            candidate.model.get('global_shift', {'mode': 'off'})),
        'p2_detail: absent/disabled -> {}'.format(
            candidate.model.get('p2_detail', {'enabled': False})),
        '',
        'NON_STRUCTURAL_PROTOCOL_DIFF: {}'.format(
            'NONE' if not protocol_diffs else ', '.join(protocol_diffs)),
    ])
    return '\n'.join(lines) + '\n', protocol_diffs


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Phase 3C preflight')
    device = torch.device('cuda:0')
    set_random_seed(0, deterministic=True)

    cfg = Config.fromfile(str(CONFIG_PATH))
    baseline_cfg = Config.fromfile(str(BASELINE_CONFIG_PATH))
    cfg.model.backbone.pretrained = cfg.model.backbone.pretrained
    pretrain_path = ROOT / cfg.model.backbone.pretrained
    if not pretrain_path.exists():
        raise FileNotFoundError(str(pretrain_path))

    resolved_config_path = ROOT / 'experiments' / 'phase3c_resolved_config.py'
    cfg.dump(str(resolved_config_path))
    diff_text, protocol_diffs = _protocol_diff(baseline_cfg, cfg)
    DIFF_PATH.write_text(diff_text)

    model = _build(cfg)
    p2_keys = sorted(
        key for key in model.state_dict()
        if key.startswith('neck.lateral_convs.0.') or
        key.startswith('neck.fpn_convs.0.'))
    fusion_types = _effective_fusion_types(cfg)
    audit = {
        'config': str(CONFIG_PATH),
        'resolved_config': str(resolved_config_path),
        'seed': cfg.seed,
        'deterministic': cfg.deterministic,
        'backbone_pretrained_field': cfg.model.backbone.pretrained,
        'backbone_pretrained_source': str(pretrain_path),
        'backbone_pretrained_sha256': _sha256(pretrain_path),
        'load_from': cfg.get('load_from'),
        'resume_from': cfg.get('resume_from'),
        'model_state_key_count': len(model.state_dict()),
        'p2_fresh_init_confirmation': {
            'p2_detail_enabled': bool(
                cfg.model.get('p2_detail', {}).get('enabled', False)),
            'spdi_module_present': model.p2_detail_injection is not None,
            'fresh_fpn_p2_keys': p2_keys,
            'start_level': cfg.model.neck.start_level,
        },
        'fusion_types': fusion_types,
        'fusion_runtime': _effective_attention_mode(model),
        'p3_p4_p5_full_attention': all(
            name == 'Fusion' and mode == 'full'
            for name, mode in _effective_attention_mode(model)[1:4]),
        'global_shift': {
            'config': cfg.model.get('global_shift', {'mode': 'off'}),
            'runtime_mode': model.global_thermal_shift.mode,
        },
        'spdi_disabled': model.p2_detail_injection is None,
        'detection_strides': list(
            cfg.model.bbox_head.anchor_generator.strides),
        'fpn_start_level': cfg.model.neck.start_level,
        'fpn_num_outs': cfg.model.neck.num_outs,
        'formal_training_from_baseline_checkpoint': False,
        'semantic_warmstart_used': False,
        'protocol_diff_names': protocol_diffs,
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2))

    model = model.to(device)
    model = MMDataParallel(model, device_ids=[0])
    dataset = build_dataset(cfg.data.train)
    model.module.CLASSES = dataset.CLASSES
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=8,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=True)
    optimizer = build_optimizer(model, cfg.optimizer)

    metrics = []
    status = 'PASS'
    error = None
    peak_driver = 0
    iterator = iter(data_loader)
    processes_before = _gpu_processes()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start_total = time.time()
    try:
        for iteration in range(5):
            data = next(iterator)
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            outputs = model.train_step(data, optimizer)
            outputs['loss'].backward()
            grad_cfg = cfg.optimizer_config.get('grad_clip')
            if grad_cfg is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_cfg['max_norm'],
                    norm_type=grad_cfg.get('norm_type', 2))
            optimizer.step()
            torch.cuda.synchronize(device)
            driver = _driver_used_mb()
            peak_driver = max(peak_driver, driver or 0)
            log_vars = {
                key: float(value)
                for key, value in outputs['log_vars'].items()}
            if not torch.isfinite(outputs['loss']).item():
                raise RuntimeError('non-finite total loss')
            if any(not torch.isfinite(torch.tensor(value)).item()
                   for value in log_vars.values()):
                raise RuntimeError('non-finite loss component')
            metrics.append({
                'iteration': iteration + 1,
                'log_vars': log_vars,
                'loss': float(outputs['loss'].detach().cpu()),
                'seconds': time.time() - start,
                'driver_used_mb': driver,
            })
    except RuntimeError as exc:
        status = 'OOM' if 'out of memory' in str(exc).lower() else 'FAIL'
        error = repr(exc)

    preflight = {
        'status': status if len(metrics) == 5 else 'FAIL',
        'config': str(CONFIG_PATH),
        'initialization_audit': str(AUDIT_PATH),
        'iterations_requested': 5,
        'iterations_completed': len(metrics),
        'batch_size': 8,
        'input_resolution': [512, 640],
        'amp': False,
        'gradient_accumulation': False,
        'metrics': metrics,
        'total_seconds': time.time() - start_total,
        'mean_iteration_seconds': (
            sum(item['seconds'] for item in metrics) / len(metrics)
            if metrics else None),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'peak_driver_used_mb': peak_driver,
        'gpu_processes_before': processes_before,
        'error': error,
    }
    PREFLIGHT_PATH.write_text(json.dumps(preflight, indent=2))
    print(json.dumps({'audit': audit, 'preflight': preflight}, indent=2))
    if preflight['status'] != 'PASS':
        raise SystemExit(2 if status == 'OOM' else 1)


if __name__ == '__main__':
    main()
