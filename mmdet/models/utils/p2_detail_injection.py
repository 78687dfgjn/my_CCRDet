"""Selective injection of lightweight C2/P2 detail into the P3 feature.

The branch is deliberately separate from the released FPN.  It is only built
when explicitly enabled in the detector config, so the disabled path has no
additional state-dict entries or numerical operations.
"""

import torch
import torch.nn as nn


class P2DetailInjection(nn.Module):
    """Fuse C2/P2 modalities, downsample, and inject detail into P3.

    Args:
        in_channels: Channel count of C2/P2 and P3 features.
        gate_channels: 1 for a spatial gate, or ``in_channels`` for a
            channel-wise gate.
        zero_init: Initialize the residual scale ``alpha`` to zero.
    """

    def __init__(self, in_channels=256, gate_channels=1, zero_init=True,
                 init_mode=None):
        super().__init__()
        if gate_channels not in (1, in_channels):
            raise ValueError(
                'gate_channels must be 1 or in_channels, got {}'.format(
                    gate_channels))

        # This is the P2 Fusion_CAT operation: concat followed by 1x1 mixing.
        self.p2_fusion = nn.Conv2d(2 * in_channels, in_channels, 1)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, in_channels, 1)
        self.gate = nn.Conv2d(2 * in_channels, gate_channels, 1)
        # ``zero_alpha`` is the original SPDI-v1 parameterization.  The
        # ``zero_projection`` mode preserves exact identity while allowing
        # useful gradients through the detail branch from the first update.
        if init_mode is None:
            init_mode = 'zero_alpha' if zero_init else 'standard_alpha'
        init_mode = str(init_mode).lower()
        if init_mode not in ('zero_alpha', 'zero_projection',
                             'standard_alpha'):
            raise ValueError(
                'init_mode must be zero_alpha, zero_projection, or '
                'standard_alpha, got {}'.format(init_mode))
        self.alpha = nn.Parameter(
            torch.zeros(1) if init_mode == 'zero_alpha' else torch.ones(1))
        if init_mode == 'zero_projection':
            nn.init.zeros_(self.pointwise.weight)
            nn.init.zeros_(self.pointwise.bias)
        self.gate_channels = gate_channels
        self.zero_init = bool(zero_init)
        self.init_mode = init_mode

    def build_detail(self, p2_rgb, p2_thermal):
        """Return the P2 detail tensor at the P3 spatial resolution."""
        p2_fused = self.p2_fusion(torch.cat((p2_rgb, p2_thermal), dim=1))
        return self.pointwise(self.depthwise(p2_fused))

    def forward(self, p3_base, p2_rgb, p2_thermal):
        detail = self.build_detail(p2_rgb, p2_thermal)
        if detail.shape[-2:] != p3_base.shape[-2:]:
            raise RuntimeError(
                'SPDI spatial mismatch: detail={} vs p3={}'.format(
                    tuple(detail.shape), tuple(p3_base.shape)))
        gate_input = torch.cat((p3_base, detail), dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        return p3_base + self.alpha * gate * detail
