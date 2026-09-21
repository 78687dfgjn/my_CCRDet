"""Lightweight groupwise complementary fusion for the P2 level.

The module preserves the Fusion_CAT convolution name so a semantic Lite-P2
checkpoint can warm-start the shared final projection directly.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class P2GroupwiseComplementaryFusion(nn.Module):
    """P2 RGB-T fusion with local groupwise modality competition."""

    def __init__(self, in_channels=256, groups=16, analysis_enabled=False,
                 checkpoint_enabled=False):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        self.in_channels = int(in_channels)
        self.groups = int(groups)
        self.channels_per_group = self.in_channels // self.groups
        self.analysis_enabled = bool(analysis_enabled)
        self.checkpoint_enabled = bool(checkpoint_enabled)
        self.conv1x1 = nn.Conv2d(2 * self.in_channels, self.in_channels, 1)
        self.reduce = nn.Conv2d(3 * self.in_channels, 32, 1)
        self.depthwise = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.relu = nn.ReLU(inplace=True)
        self.gate_logits = nn.Conv2d(32, 2 * self.groups, 1)
        nn.init.zeros_(self.gate_logits.weight)
        nn.init.zeros_(self.gate_logits.bias)
        self.reset_analysis()

    def reset_analysis(self):
        self._analysis = {
            'calls': 0,
            'count': 0,
            'rgb_sum': 0.0,
            'thermal_sum': 0.0,
            'rgb_dev_sq_sum': 0.0,
            'thermal_dev_sq_sum': 0.0,
            'abs_dev_sum': 0.0,
            'rgb_gt_125': 0,
            'thermal_gt_125': 0,
            'min_rgb': float('inf'),
            'max_rgb': float('-inf'),
            'min_thermal': float('inf'),
            'max_thermal': float('-inf'),
            'group_rgb_sum': [0.0] * self.groups,
            'group_thermal_sum': [0.0] * self.groups,
        }

    @torch.no_grad()
    def _record_analysis(self, weights):
        if not self.analysis_enabled:
            return
        rgb = weights[:, 0]
        thermal = weights[:, 1]
        rgb_flat = rgb.detach().float()
        thermal_flat = thermal.detach().float()
        count = int(rgb_flat.numel())
        if count == 0:
            return
        stats = self._analysis
        stats['calls'] += 1
        stats['count'] += count
        stats['rgb_sum'] += float(rgb_flat.sum().item())
        stats['thermal_sum'] += float(thermal_flat.sum().item())
        stats['rgb_dev_sq_sum'] += float(
            ((rgb_flat - 1.0) * (rgb_flat - 1.0)).sum().item())
        stats['thermal_dev_sq_sum'] += float(
            ((thermal_flat - 1.0) * (thermal_flat - 1.0)).sum().item())
        stats['abs_dev_sum'] += float(
            (rgb_flat - 1.0).abs().sum().item() +
            (thermal_flat - 1.0).abs().sum().item())
        stats['rgb_gt_125'] += int((rgb_flat > 1.25).sum().item())
        stats['thermal_gt_125'] += int((thermal_flat > 1.25).sum().item())
        stats['min_rgb'] = min(stats['min_rgb'], float(rgb_flat.min().item()))
        stats['max_rgb'] = max(stats['max_rgb'], float(rgb_flat.max().item()))
        stats['min_thermal'] = min(
            stats['min_thermal'], float(thermal_flat.min().item()))
        stats['max_thermal'] = max(
            stats['max_thermal'], float(thermal_flat.max().item()))
        for group in range(self.groups):
            stats['group_rgb_sum'][group] += float(
                rgb_flat[:, group].sum().item())
            stats['group_thermal_sum'][group] += float(
                thermal_flat[:, group].sum().item())

    def get_analysis_stats(self):
        stats = self._analysis
        count = max(int(stats['count']), 1)
        rgb_mean = stats['rgb_sum'] / count
        thermal_mean = stats['thermal_sum'] / count
        rgb_var = stats['rgb_dev_sq_sum'] / count
        thermal_var = stats['thermal_dev_sq_sum'] / count
        group_count = max(count // self.groups, 1)
        return {
            'calls': int(stats['calls']),
            'count': int(stats['count']),
            'mean_rgb_weight': rgb_mean,
            'mean_thermal_weight': thermal_mean,
            'std_rgb_weight': rgb_var ** 0.5,
            'std_thermal_weight': thermal_var ** 0.5,
            'mean_abs_weight_minus_1': stats['abs_dev_sum'] / (2 * count),
            'fraction_rgb_weight_gt_1_25': stats['rgb_gt_125'] / count,
            'fraction_thermal_weight_gt_1_25': stats['thermal_gt_125'] / count,
            'min_rgb_weight': stats['min_rgb'],
            'max_rgb_weight': stats['max_rgb'],
            'min_thermal_weight': stats['min_thermal'],
            'max_thermal_weight': stats['max_thermal'],
            'group_mean_rgb_weight': [
                value / group_count for value in stats['group_rgb_sum']],
            'group_mean_thermal_weight': [
                value / group_count for value in stats['group_thermal_sum']],
        }

    def _forward_impl(self, rgb, thermal, record_analysis):
        if rgb.shape != thermal.shape:
            raise ValueError('RGB and thermal P2 features must have equal shapes')
        batch, _, height, width = rgb.shape
        difference = torch.abs(rgb - thermal)
        gate_input = torch.cat((rgb, thermal, difference), dim=1)
        hidden = self.relu(self.reduce(gate_input))
        hidden = self.relu(self.depthwise(hidden))
        logits = self.gate_logits(hidden)
        logits = logits.view(batch, 2, self.groups, height, width)
        weights = 2.0 * torch.softmax(logits, dim=1)
        if record_analysis:
            self._record_analysis(weights)
        rgb_weight = weights[:, 0].unsqueeze(2)
        thermal_weight = weights[:, 1].unsqueeze(2)
        rgb_grouped = rgb.view(
            batch, self.groups, self.channels_per_group, height, width)
        thermal_grouped = thermal.view(
            batch, self.groups, self.channels_per_group, height, width)
        rgb_weighted = (rgb_grouped * rgb_weight).reshape(
            batch, self.in_channels, height, width)
        thermal_weighted = (thermal_grouped * thermal_weight).reshape(
            batch, self.in_channels, height, width)
        return self.conv1x1(torch.cat((rgb_weighted, thermal_weighted), dim=1))

    def _checkpoint_forward(self, rgb, thermal):
        # PyTorch 1.10-compatible reentrant checkpoint path.  This function is
        # pure with respect to telemetry, so backward recomputation is not
        # counted as a second analysis call.
        return self._forward_impl(rgb, thermal, False)

    def forward(self, rgb, thermal):
        if self.training and self.checkpoint_enabled:
            return checkpoint(self._checkpoint_forward, rgb, thermal)
        return self._forward_impl(
            rgb, thermal, self.analysis_enabled)
