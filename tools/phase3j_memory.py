import json, subprocess, time
from pathlib import Path
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.apis import set_random_seed
from mmdet.core import build_optimizer
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

CONFIG = 'configs_local/exp_proxy_pseudo_p2_topdown_seed0.py'
SOURCE = '/hy-tmp/CCRDet_assets/phase3a_semantic_warmstart_seed0.pth'
OUTPUT = 'experiments/phase3j_memory.json'

def driver_mb():
    try:
        x = subprocess.check_output(
            ['nvidia-smi','--query-gpu=memory.used',
             '--format=csv,noheader,nounits'], text=True)
        return int(x.strip().splitlines()[0])
    except Exception:
        return None

cfg = Config.fromfile(CONFIG)
cfg.model.backbone.pretrained = None
set_random_seed(0, deterministic=True)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA required')
device = torch.device('cuda:0')
dataset = build_dataset(cfg.data.train)
loader = build_dataloader(
    dataset, samples_per_gpu=8, workers_per_gpu=cfg.data.workers_per_gpu,
    num_gpus=1, dist=False, shuffle=True)
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                       test_cfg=cfg.get('test_cfg'))
model.init_weights()
source = torch.load(SOURCE, map_location='cpu')['state_dict']
incompat = model.load_state_dict(source, strict=False)
model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])
optimizer = build_optimizer(model, cfg.optimizer)
shapes = {}
hooks = []
for i, mod in enumerate(model.module.fuse):
    hooks.append(mod.register_forward_hook(
        lambda m, ins, out, i=i: shapes.update({'P%d' % (i+2): list(out.shape)})))
rows = []
status = 'PASS'
error = None
peak_driver = 0
it = iter(loader)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
try:
    for i in range(5):
        data = next(it)
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        out = model.train_step(data, optimizer)
        loss = out['loss']
        if not torch.isfinite(loss).item():
            raise FloatingPointError('non-finite loss')
        loss.backward()
        g = cfg.optimizer_config.get('grad_clip')
        if g:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=g['max_norm'],
                norm_type=g.get('norm_type', 2))
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.time() - start
        drv = driver_mb() or 0
        peak_driver = max(peak_driver, drv)
        rows.append({
            'iteration': i+1, 'loss': float(loss.detach().cpu()),
            'log_vars': {k: float(v) for k,v in out['log_vars'].items()},
            'iteration_seconds': elapsed, 'driver_used_mb': drv,
            'allocated_bytes': torch.cuda.memory_allocated(device),
            'reserved_bytes': torch.cuda.memory_reserved(device)})
except RuntimeError as exc:
    status = 'RESOURCE_BLOCKED' if 'out of memory' in str(exc).lower() else 'FAIL'
    error = repr(exc)
except Exception as exc:
    status = 'FAIL'
    error = repr(exc)
for h in hooks:
    h.remove()
report = {
    'status': status, 'config': CONFIG, 'source_checkpoint': SOURCE,
    'missing_keys': list(incompat.missing_keys),
    'unexpected_keys': list(incompat.unexpected_keys),
    'iterations_requested': 5, 'iterations_completed': len(rows),
    'batch_size': 8, 'input_resolution': [512,640],
    'amp': False, 'gradient_accumulation': False,
    'feature_shapes': shapes, 'metrics': rows,
    'mean_iteration_seconds': sum(x['iteration_seconds'] for x in rows)/len(rows) if rows else None,
    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
    'peak_driver_used_mb': peak_driver, 'error': error}
Path(OUTPUT).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if status != 'PASS':
    raise SystemExit(2)
