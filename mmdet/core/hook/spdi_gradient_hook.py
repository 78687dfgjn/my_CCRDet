"""Analysis-only gradient tracing for the SPDI paired proxy experiment."""

import json
import math
import os

import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class SPDIGradientHook(Hook):
    """Record SPDI scale and branch gradients without changing training."""

    def __init__(self, output_json, iteration_points=None):
        self.output_json = output_json
        self.iteration_points = set(iteration_points or
                                    [1, 2, 5, 10, 20, 50, 100])
        self.records = []

    @staticmethod
    def _unwrap(model):
        return getattr(model, 'module', model)

    @staticmethod
    def _grad_norm(module):
        grads = [p.grad.detach().reshape(-1) for p in module.parameters()
                 if p.grad is not None]
        if not grads:
            return None
        return float(torch.cat(grads).norm().item())

    @staticmethod
    def _scalar(tensor):
        if tensor is None:
            return None
        value = float(tensor.detach().reshape(-1)[0].item())
        return value if math.isfinite(value) else None

    def _capture(self, runner, label):
        model = self._unwrap(runner.model)
        module = getattr(model, 'p2_detail_injection', None)
        if module is None:
            raise RuntimeError('SPDIGradientHook requires an enabled SPDI module')

        record = dict(
            label=label,
            epoch=int(runner.epoch) + 1,
            iteration=int(runner.iter) + 1,
            alpha=self._scalar(module.alpha),
            alpha_grad=self._scalar(module.alpha.grad),
            p2_fusion_grad_norm=self._grad_norm(module.p2_fusion),
            depthwise_grad_norm=self._grad_norm(module.depthwise),
            pointwise_grad_norm=self._grad_norm(module.pointwise),
            gate_grad_norm=self._grad_norm(module.gate),
        )
        record['all_finite'] = all(
            value is None or math.isfinite(float(value))
            for key, value in record.items()
            if key not in ('label', 'epoch', 'iteration'))
        self.records.append(record)

    def after_train_iter(self, runner):
        iteration = int(runner.iter) + 1
        if iteration in self.iteration_points:
            self._capture(runner, 'iter_{}'.format(iteration))

    def after_train_epoch(self, runner):
        self._capture(runner, 'epoch_{}_end'.format(int(runner.epoch) + 1))

    def after_run(self, runner):
        directory = os.path.dirname(os.path.abspath(self.output_json))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = dict(
            experiment='phase3d_spdi_paired_proxy',
            iteration_points=sorted(self.iteration_points),
            records=self.records,
        )
        with open(self.output_json, 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
