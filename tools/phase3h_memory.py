"""20-iteration full-resolution PGCF memory and finite-loss smoke."""
import json
import subprocess
import time
from pathlib import Path

import torch
from mmcv.parallel import MMDataParallel
from mmdet.apis import set_random_seed
from mmdet.core import build_optimizer
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


CONFIG = 'configs_local/exp_proxy_pgcf_p2p6_seed0.py'
SOURCE = '/hy-tmp/CCRDet_assets/phase3a_semantic_warmstart_seed0.pth'


def driver_used_mb():
    try:
        text = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.STDOUT, text=True)
        return int(text.strip().splitlines()[0])
    except Exception:
        return None


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the memory smoke')
    from mmcv import Config
    cfg = Config.fromfile(CONFIG)
    cfg.model.backbone.pretrained = None
    set_random_seed(0, deterministic=True)
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(
        dataset, samples_per_gpu=8,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1, dist=False, shuffle=True)
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    source = torch.load(SOURCE, map_location='cpu')['state_dict']
    incompat = model.load_state_dict(source, strict=False)
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    optimizer = build_optimizer(model, cfg.optimizer)
    pgcf = model.module.fuse[0]
    pgcf.analysis_enabled = True
    feature_shapes = {}
    hook = pgcf.register_forward_hook(
        lambda module, inputs, output: feature_shapes.update(
            {'output': list(output.shape)}))
    device = torch.device('cuda:0')
    iterator = iter(loader)
    rows = []
    times = []
    peak_driver = 0
    status = 'PASS'
    error = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for iteration in range(20):
            data = next(iterator)
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            outputs = model.train_step(data, optimizer)
            loss = outputs['loss']
            if not torch.isfinite(loss).item():
                raise FloatingPointError('non-finite total loss')
            loss.backward()
            grad_cfg = cfg.optimizer_config.get('grad_clip')
            if grad_cfg is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_cfg['grad_clip']['max_norm']
                    if isinstance(grad_cfg, dict) and
                    'grad_clip' in grad_cfg else grad_cfg['max_norm'],
                    norm_type=grad_cfg.get('norm_type', 2))
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.time() - start
            times.append(elapsed)
            peak_driver = max(peak_driver, driver_used_mb() or 0)
            rows.append({
                'iteration': iteration + 1,
                'loss': float(loss.detach().cpu()),
                'log_vars': {
                    key: float(value)
                    for key, value in outputs['log_vars'].items()},
                'iteration_seconds': elapsed,
                'driver_used_mb': driver_used_mb(),
            })
    except RuntimeError as exc:
        status = ('RESOURCE_BLOCKED' if 'out of memory' in str(exc).lower()
                  else 'FAIL')
        error = repr(exc)
    except Exception as exc:
        status = 'FAIL'
        error = repr(exc)
    finally:
        hook.remove()
    report = {
        'status': status,
        'config': CONFIG,
        'source_checkpoint': SOURCE,
        'missing_keys': list(incompat.missing_keys),
        'unexpected_keys': list(incompat.unexpected_keys),
        'iterations_requested': 20,
        'iterations_completed': len(rows),
        'batch_size': 8,
        'input_resolution': [512, 640],
        'amp': False,
        'gradient_accumulation': False,
        'feature_shapes': feature_shapes,
        'metrics': rows,
        'mean_iteration_seconds': (
            sum(times) / len(times) if times else None),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'peak_driver_used_mb': peak_driver,
        'gate_stats_at_smoke_end': pgcf.get_analysis_stats(),
        'error': error,
    }
    Path('experiments').mkdir(exist_ok=True)
    Path('experiments/phase3h_memory.json').write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if status != 'PASS':
        raise SystemExit(2 if status == 'RESOURCE_BLOCKED' else 1)


if __name__ == '__main__':
    main()
