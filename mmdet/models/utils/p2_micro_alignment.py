"""Bounded P2-only residual thermal feature alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class P2MicroAlignment(nn.Module):
    """Predict a bounded input-pixel residual and warp thermal P2 only.

    The returned thermal feature is sampled with base - displacement.
    Thus a negative dx means thermal content moves left, matching the
    GlobalThermalShift convention. Telemetry is stored as plain attributes
    and never enters state_dict.
    """

    def __init__(self, modality_channels=256, hidden_channels=64,
                 feature_stride=4, max_residual_px=1.0,
                 analysis_enabled=False):
        super().__init__()
        self.modality_channels = int(modality_channels)
        self.feature_stride = float(feature_stride)
        self.max_residual_px = float(max_residual_px)
        if self.feature_stride <= 0 or self.max_residual_px <= 0:
            raise ValueError('feature_stride and max_residual_px must be > 0')
        self.reduce = nn.Conv2d(2 * self.modality_channels,
                                int(hidden_channels), kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.offset = nn.Conv2d(int(hidden_channels), 2, kernel_size=3,
                                padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.analysis_enabled = bool(analysis_enabled)
        self.last_stats = {}

    def _base_grid(self, batch, height, width, device, dtype):
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5)
        xs = xs * (2.0 / float(width)) - 1.0
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5)
        ys = ys * (2.0 / float(height)) - 1.0
        yy, xx = torch.meshgrid(ys, xs)
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        return grid.expand(batch, -1, -1, -1)

    def _record_stats(self, delta):
        with torch.no_grad():
            dx = delta[:, 0]
            dy = delta[:, 1]
            abs_dx = dx.abs()
            abs_dy = dy.abs()
            self.last_stats = {
                'mean_dx': float(dx.mean().item()),
                'mean_dy': float(dy.mean().item()),
                'std_dx': float(dx.std(unbiased=False).item()),
                'std_dy': float(dy.std(unbiased=False).item()),
                'mean_abs_dx': float(abs_dx.mean().item()),
                'mean_abs_dy': float(abs_dy.mean().item()),
                'max_abs_dx': float(abs_dx.max().item()),
                'max_abs_dy': float(abs_dy.max().item()),
                'saturation_ratio_dx': float((abs_dx > 0.9).float().mean().item()),
                'saturation_ratio_dy': float((abs_dy > 0.9).float().mean().item()),
                'units': 'input_image_pixels',
            }

    def forward(self, rgb_p2, thermal_p2):
        if rgb_p2.shape != thermal_p2.shape:
            raise ValueError('RGB and thermal P2 shapes must match')
        batch, _, height, width = thermal_p2.shape
        fused = torch.cat((rgb_p2, thermal_p2), dim=1)
        raw = self.offset(self.relu(self.reduce(fused)))
        delta = self.max_residual_px * torch.tanh(raw)

        grid = self._base_grid(batch, height, width, thermal_p2.device,
                               thermal_p2.dtype).clone()
        grid[..., 0] = grid[..., 0] - (
            2.0 * delta[:, 0] / (self.feature_stride * float(width)))
        grid[..., 1] = grid[..., 1] - (
            2.0 * delta[:, 1] / (self.feature_stride * float(height)))
        warped = F.grid_sample(
            thermal_p2, grid, mode='bilinear', padding_mode='zeros',
            align_corners=False)
        if self.analysis_enabled:
            self._record_stats(delta)
        return warped
