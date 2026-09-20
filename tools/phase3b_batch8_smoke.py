"""Run a faithful batch-8 full-resolution train-step smoke test."""
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
    config_path = 'configs_local/exp_p2p6_seed0.py'
    checkpoint_path = '/hy-tmp/CCRDet_assets/phase3a_semantic_warmstart_seed0.pth'
    report_path = Path('experiments/phase3b_batch8_smoke.json')
    cfg = Config.fromfile(config_path)
    cfg.model.backbone.pretrained = None
    set_random_seed(0, deterministic=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the batch-8 smoke test')
    device = torch.device('cuda:0')
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=8,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=True)
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True)
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.to(device), device_ids=[0])
    optimizer = build_optimizer(model, cfg.optimizer)

    feature_shapes = {}
    hooks = []
    for index, module in enumerate(model.module.fuse):
        def save_shape(_module, _inputs, output, index=index):
            feature_shapes[index] = list(output.shape)
        hooks.append(module.register_forward_hook(save_shape))

    times = []
    metrics = []
    peak_driver_mb = 0
    iterator = iter(data_loader)
    status = 'PASS'
    error = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(20):
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
            times.append(time.time() - start)
            peak_driver_mb = max(peak_driver_mb, driver_used_mb() or 0)
            metrics.append({
                'iteration': iteration + 1,
                'log_vars': {k: float(v) for k, v in outputs['log_vars'].items()},
                'loss': float(outputs['loss'].detach().cpu()),
                'iteration_seconds': times[-1],
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
        'config': config_path,
        'checkpoint': checkpoint_path,
        'iterations_requested': 20,
        'iterations_completed': len(metrics),
        'batch_size': 8,
        'input_resolution': [512, 640],
        'amp': False,
        'gradient_accumulation': False,
        'feature_shapes': {str(k): v for k, v in sorted(feature_shapes.items())},
        'metrics': metrics,
        'mean_iteration_seconds': (sum(times) / len(times) if times else None),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'peak_driver_used_mb': peak_driver_mb,
        'error': error,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if status != 'PASS':
        raise SystemExit(2 if status == 'OOM' else 1)


if __name__ == '__main__':
    main()
