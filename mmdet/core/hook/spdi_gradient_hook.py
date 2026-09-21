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


@HOOKS.register_module()
class SPDIV2DynamicsHook(Hook):
    """Trace SPDI-v2 optimization and gate continuation at iteration 100.

    The hook is analysis-only until the first 100-iteration precheck is
    complete.  If the exact gradient/parameter conditions are not observed,
    the run is stopped rather than silently continuing to a proxy result.
    """

    def __init__(self, output_json, iteration_points=None):
        self.output_json = output_json
        self.iteration_points = set(iteration_points or
                                    [1, 2, 5, 10, 20, 50, 100])
        self.records = []
        self.precheck_status = 'RUNNING'
        self.precheck_failure = None

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
    def _norm(parameter):
        if parameter is None:
            return None
        return float(parameter.detach().norm().item())

    @staticmethod
    def _scalar(tensor):
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            value = float(tensor)
        else:
            value = float(tensor.detach().reshape(-1)[0].item())
        return value if math.isfinite(value) else None

    def _capture(self, runner, label):
        model = self._unwrap(runner.model)
        module = getattr(model, 'p2_detail_injection', None)
        if module is None:
            raise RuntimeError('SPDIV2DynamicsHook requires enabled SPDI')
        loss = None
        if isinstance(getattr(runner, 'outputs', None), dict):
            loss = runner.outputs.get('loss')
        record = dict(
            label=label,
            epoch=int(runner.epoch) + 1,
            iteration=int(runner.iter) + 1,
            alpha=self._scalar(module.alpha),
            alpha_grad=self._scalar(module.alpha.grad),
            pointwise_weight_norm=self._norm(module.pointwise.weight),
            pointwise_grad_norm=self._grad_norm(module.pointwise),
            depthwise_grad_norm=self._grad_norm(module.depthwise),
            p2_fusion_grad_norm=self._grad_norm(module.p2_fusion),
            gate_grad_norm=self._grad_norm(module.gate),
            loss=self._scalar(loss),
        )
        record['all_finite'] = all(
            value is None or math.isfinite(float(value))
            for key, value in record.items()
            if key not in ('label', 'epoch', 'iteration'))
        self.records.append(record)
        return record

    def _payload(self):
        return dict(
            experiment='phase3e_spdi_v2_optimization_precheck',
            iteration_points=sorted(self.iteration_points),
            precheck_status=self.precheck_status,
            precheck_failure=self.precheck_failure,
            records=self.records,
        )

    def _write(self):
        directory = os.path.dirname(os.path.abspath(self.output_json))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.output_json, 'w') as handle:
            json.dump(self._payload(), handle, indent=2, sort_keys=True)

    def after_train_iter(self, runner):
        iteration = int(runner.iter) + 1
        if iteration not in self.iteration_points:
            return
        self._capture(runner, 'iter_{}'.format(iteration))
        if iteration != 100:
            return
        by_label = {r['label']: r for r in self.records}
        r1 = by_label.get('iter_1', {})
        r2 = by_label.get('iter_2', {})
        r100 = by_label.get('iter_100', {})
        failures = []
        if not all(r.get('all_finite', False) for r in self.records):
            failures.append('non-finite dynamics record')
        if not (r1.get('pointwise_grad_norm') is not None and
                r1['pointwise_grad_norm'] > 0.0):
            failures.append('iter1 pointwise gradient is zero or missing')
        if not (r2.get('depthwise_grad_norm') is not None and
                r2['depthwise_grad_norm'] > 0.0):
            failures.append('iter2 depthwise gradient is zero or missing')
        if not (r2.get('p2_fusion_grad_norm') is not None and
                r2['p2_fusion_grad_norm'] > 0.0):
            failures.append('iter2 p2_fusion gradient is zero or missing')
        if not (r100.get('pointwise_weight_norm') is not None and
                r100['pointwise_weight_norm'] > 0.0):
            failures.append('iter100 pointwise weight did not leave zero')
        if failures:
            self.precheck_status = 'FAIL'
            self.precheck_failure = failures
            self._write()
            raise RuntimeError('SPDI-V2 OPTIMIZATION FAILURE: {}'.format(
                '; '.join(failures)))
        self.precheck_status = 'PASS'
        self._write()

    def after_train_epoch(self, runner):
        self._capture(runner, 'epoch_{}_end'.format(int(runner.epoch) + 1))
        self._write()

    def after_run(self, runner):
        if self.precheck_status == 'RUNNING':
            self.precheck_status = 'INCOMPLETE'
        self._write()
