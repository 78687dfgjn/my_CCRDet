"""Analysis-only telemetry hook for the Phase 3F P2 alignment proxy."""

import json
import math
import os
import subprocess

import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class Phase3FAlignmentHook(Hook):

    def __init__(self, output_json, iteration_points=None):
        self.output_json = output_json
        self.iteration_points = set(iteration_points or
                                    [1, 2, 5, 10, 20, 50, 100])
        self.records = []

    @staticmethod
    def _unwrap(model):
        return getattr(model, 'module', model)

    @staticmethod
    def _norm(module):
        if module is None:
            return None
        values = [p.grad.detach().reshape(-1) for p in module.parameters()
                  if p.grad is not None]
        return float(torch.cat(values).norm().item()) if values else None

    @staticmethod
    def _param_norm(module):
        if module is None:
            return None
        values = [p.detach().reshape(-1) for p in module.parameters()]
        return float(torch.cat(values).norm().item()) if values else None

    def _capture(self, runner, label):
        model = self._unwrap(runner.model)
        module = getattr(model, 'p2_micro_alignment', None)
        if module is None:
            raise RuntimeError('Phase3FAlignmentHook requires P2 alignment')
        outputs = getattr(runner, 'outputs', {})
        loss = outputs.get('loss') if isinstance(outputs, dict) else None
        stats = dict(getattr(module, 'last_stats', {}))
        driver_used_mb = None
        if torch.cuda.is_available():
            try:
                output = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.used',
                     '--format=csv,noheader,nounits'],
                    text=True).strip().splitlines()
                driver_used_mb = int(float(output[0].strip()))
            except Exception:
                driver_used_mb = None
        record = dict(
            label=label,
            epoch=int(runner.epoch) + 1,
            iteration=int(runner.iter) + 1,
            loss=(float(loss.detach().item()) if torch.is_tensor(loss)
                  else None),
            allocated_bytes=(int(torch.cuda.memory_allocated())
                             if torch.cuda.is_available() else None),
            reserved_bytes=(int(torch.cuda.memory_reserved())
                            if torch.cuda.is_available() else None),
            max_allocated_bytes=(int(torch.cuda.max_memory_allocated())
                                 if torch.cuda.is_available() else None),
            max_reserved_bytes=(int(torch.cuda.max_memory_reserved())
                                if torch.cuda.is_available() else None),
            driver_used_mb=driver_used_mb,
            offset_grad_norm=self._norm(module.offset),
            reduce_grad_norm=self._norm(module.reduce),
            offset_weight_norm=self._param_norm(module.offset),
            reduce_weight_norm=self._param_norm(module.reduce),
            telemetry=stats,
        )
        numeric = [v for v in record.values() if isinstance(v, (int, float))]
        record['all_finite'] = all(math.isfinite(float(v))
                                   for v in numeric if v is not None)
        self.records.append(record)

    def after_train_iter(self, runner):
        iteration = int(runner.iter) + 1
        if iteration in self.iteration_points:
            self._capture(runner, 'iter_{}'.format(iteration))

    def after_train_epoch(self, runner):
        self._capture(runner, 'epoch_{}_end'.format(int(runner.epoch) + 1))
        self._write()

    def _write(self):
        parent = os.path.dirname(os.path.abspath(self.output_json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.output_json, 'w') as f:
            json.dump({
                'experiment': 'phase3f_p2_alignment_dynamics',
                'iteration_points': sorted(self.iteration_points),
                'records': self.records,
            }, f, indent=2, sort_keys=True)

    def after_run(self, runner):
        self._write()
