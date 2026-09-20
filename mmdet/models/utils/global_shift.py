"""Differentiable global translation for the thermal stream.

The translation parameters are expressed in input-image pixels. A negative
``dx`` means that the thermal content moves to the left in the output.
"""

import torch
import torch.nn.functional as F
from torch import nn


class GlobalThermalShift(nn.Module):
    """Apply an optional global translation to a thermal image tensor."""

    def __init__(self, mode='off', dx=0.0, dy=0.0, max_shift=2.0):
        super().__init__()
        mode = str(mode).lower()
        if mode not in ('off', 'fixed', 'learnable'):
            raise ValueError('mode must be off, fixed, or learnable')
        if max_shift <= 0:
            raise ValueError('max_shift must be positive')

        self.mode = mode
        self.dx = float(dx)
        self.dy = float(dy)
        self.max_shift = float(max_shift)

        if self.mode == 'learnable':
            init = torch.tensor([self.dx, self.dy], dtype=torch.float32)
            init = torch.clamp(init / self.max_shift, -0.999999, 0.999999)
            self.raw_shift = nn.Parameter(torch.atanh(init))

    def displacement(self, thermal):
        if self.mode == 'learnable':
            shift = self.max_shift * torch.tanh(self.raw_shift)
            shift = shift.to(device=thermal.device, dtype=thermal.dtype)
            return shift[0], shift[1]
        return thermal.new_tensor(self.dx), thermal.new_tensor(self.dy)

    def forward(self, thermal):
        if self.mode == 'off':
            return thermal
        if thermal.ndim != 4:
            raise ValueError('thermal must have shape [N, C, H, W]')

        dx, dy = self.displacement(thermal)
        if self.mode == 'fixed' and self.dx == 0.0 and self.dy == 0.0:
            return thermal

        batch, _, height, width = thermal.shape
        one = thermal.new_tensor(1.0)
        zero = thermal.new_tensor(0.0)
        # affine_grid maps output coordinates to input coordinates. Sampling
        # at -dx/-dy therefore moves content by dx/dy pixels.
        tx = -2.0 * dx / float(width)
        ty = -2.0 * dy / float(height)
        theta = torch.stack((one, zero, tx, zero, one, ty)).view(2, 3)
        theta = theta.unsqueeze(0).expand(batch, -1, -1)
        grid = F.affine_grid(theta, thermal.size(), align_corners=False)
        return F.grid_sample(
            thermal,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
