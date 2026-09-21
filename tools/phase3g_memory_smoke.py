"""Full-resolution Phase 3G memory/shape smoke test."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet.core import build_optimizer
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.backbone.pretrained = None
    set_random_seed(0, deterministic=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.device('cuda:0')
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(
        dataset, samples_per_gpu=8,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1, dist=False, shuffle=True)
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.to(device), device_ids=[0])
    optimizer = build_optimizer(model, cfg.optimizer)
    feature_shapes = {}
    fusion_paths = []
    hooks = []
    for index, module in enumerate(model.module.fuse):
        def save_shape(mod, inputs, output, index=index):
            feature_shapes[str(index)] = list(output.shape)
            fusion_paths.append({
                'index': index,
                'class': mod.__class__.__name__,
                'spatial': [int(output.shape[-2]), int(output.shape[-1])],
            })
        hooks.append(module.register_forward_hook(save_shape))
    metrics = []
    times = []
    peak_driver = 0
    status = 'PASS'
    error = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        iterator = iter(loader)
        for iteration in range(args.iterations):
            data = next(iterator)
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            outputs = model.train_step(data, optimizer)
            outputs['loss'].backward()
            grad_cfg = cfg.optimizer_config.get('grad_clip')
            if grad_cfg is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_cfg['max_norm'],
                    grad_cfg.get('norm_type', 2))
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.time() - start
            times.append(elapsed)
            peak_driver = max(peak_driver, driver_used_mb() or 0)
            metrics.append({
                'iteration': iteration + 1,
                'loss': float(outputs['loss'].detach().cpu()),
                'log_vars': {
                    k: float(v) for k, v in outputs['log_vars'].items()
                },
                'seconds': elapsed,
                'driver_used_mb': driver_used_mb(),
            })
    except RuntimeError as exc:
        status = 'OOM' if 'out of memory' in str(exc).lower() else 'FAIL'
        error = repr(exc)
    finally:
        for hook in hooks:
            hook.remove()
    result = {
        'status': status,
        'label': args.label,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'iterations_requested': args.iterations,
        'iterations_completed': len(metrics),
        'batch_size': 8,
        'input_resolution': [512, 640],
        'amp': False,
        'gradient_accumulation': False,
        'feature_shapes': feature_shapes,
        'fusion_paths': fusion_paths,
        'metrics': metrics,
        'mean_iteration_seconds': (
            sum(times) / len(times) if times else None),
        'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
        'peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
        'peak_driver_used_mb': peak_driver,
        'error': error,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, indent=2))
    print(json.dumps({
        k: result[k] for k in (
            'status', 'label', 'iterations_completed',
            'feature_shapes', 'fusion_paths',
            'peak_allocated_bytes', 'peak_reserved_bytes',
            'peak_driver_used_mb', 'mean_iteration_seconds', 'error')
    }, indent=2))
    if status != 'PASS':
        raise SystemExit(2 if status == 'OOM' else 1)


if __name__ == '__main__':
    main()
