# Copyright (c) OpenMMLab. All rights reserved.
"""Exact paired thermal integer translation for train-derived calibration."""

import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class PairThermalCalibration:
    """Translate only the thermal image with integer pixel semantics.

    A source pixel at (x, y) is written to (x + dx, y + dy).
    Therefore dx < 0 moves thermal content to the left and dy < 0 moves it
    upward. The transform is intentionally placed before RandomFlip so the
    calibration is applied in the original orientation.
    """

    def __init__(self, dx=0, dy=0, fill_value=93.1):
        self.dx = int(dx)
        self.dy = int(dy)
        self.fill_value = float(fill_value)

    def __call__(self, results):
        if self.dx == 0 and self.dy == 0:
            return results

        if 'img2' not in results:
            raise KeyError('PairThermalCalibration requires results["img2"]')
        thermal = results['img2']
        if not isinstance(thermal, np.ndarray):
            raise TypeError('img2 must be a numpy.ndarray')
        if thermal.ndim not in (2, 3):
            raise ValueError('img2 must be HxW or HxWxC')

        h, w = thermal.shape[:2]
        fill = self.fill_value
        if np.issubdtype(thermal.dtype, np.integer):
            fill = np.asarray(np.rint(fill), dtype=thermal.dtype).item()
        else:
            fill = np.asarray(fill, dtype=thermal.dtype).item()
        shifted = np.full_like(thermal, fill)

        src_y0 = max(0, -self.dy)
        src_y1 = min(h, h - self.dy)
        src_x0 = max(0, -self.dx)
        src_x1 = min(w, w - self.dx)
        dst_y0 = max(0, self.dy)
        dst_y1 = min(h, h + self.dy)
        dst_x0 = max(0, self.dx)
        dst_x1 = min(w, w + self.dx)

        if src_y1 > src_y0 and src_x1 > src_x0:
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = thermal[
                src_y0:src_y1, src_x0:src_x1]
        results['img2'] = shifted
        return results

    def __repr__(self):
        return (self.__class__.__name__ + '(dx={}, dy={}, fill_value={})'.format(
            self.dx, self.dy, self.fill_value))
