"""Unit tests for train-derived exact paired thermal calibration."""

import copy
import numpy as np

from mmdet.datasets.pipelines.paired_thermal_shift import PairThermalCalibration


def _flip_horizontal(results):
    out = copy.deepcopy(results)
    out['img1'] = out['img1'][:, ::-1].copy()
    out['img2'] = out['img2'][:, ::-1].copy()
    return out


def test_identity_and_protection():
    rgb = np.arange(20, dtype=np.uint8).reshape(4, 5)
    thermal = (rgb + 30).copy()
    boxes = np.array([[1, 1, 3, 3]], dtype=np.float32)
    results = {'img1': rgb.copy(), 'img2': thermal.copy(),
               'gt_bboxes': boxes.copy()}
    out = PairThermalCalibration(0, 0)(results)
    assert out is results
    assert np.array_equal(out['img1'], rgb)
    assert np.array_equal(out['img2'], thermal)
    assert np.array_equal(out['gt_bboxes'], boxes)


def test_negative_x_moves_dot_left():
    thermal = np.zeros((5, 7), dtype=np.uint8)
    thermal[2, 3] = 255
    out = PairThermalCalibration(dx=-1, dy=0, fill_value=93)(
        {'img2': thermal.copy()})
    assert out['img2'][2, 2] == 255
    assert out['img2'][2, 3] == 0
    assert out['img2'][0, 6] == 93


def test_y_sign_and_rgb_box_invariance():
    rgb = np.arange(35, dtype=np.uint8).reshape(5, 7)
    thermal = np.zeros_like(rgb)
    thermal[2, 3] = 255
    boxes = np.array([[1, 1, 4, 4]], dtype=np.float32)
    out = PairThermalCalibration(dx=0, dy=-1, fill_value=93)(
        {'img1': rgb.copy(), 'img2': thermal.copy(),
         'gt_bboxes': boxes.copy()})
    assert out['img2'][1, 3] == 255
    assert np.array_equal(out['img1'], rgb)
    assert np.array_equal(out['gt_bboxes'], boxes)


def test_shift_then_flip_equivalence():
    base = np.zeros((4, 7), dtype=np.uint8)
    base[2, 3] = 255
    pre = PairThermalCalibration(dx=-1, dy=0, fill_value=0)(
        {'img1': base.copy(), 'img2': base.copy()})
    flipped_after = _flip_horizontal(pre)
    flipped = {'img1': base[:, ::-1].copy(), 'img2': base[:, ::-1].copy()}
    # Applying the original-orientation shift before the flip is equal to
    # applying the sign-reversed shift after the flip.
    post = PairThermalCalibration(dx=1, dy=0, fill_value=0)(flipped)
    assert np.array_equal(flipped_after['img1'], post['img1'])
    assert np.array_equal(flipped_after['img2'], post['img2'])


def test_float_fill_and_normalization_nonborder():
    thermal = np.zeros((3, 5, 1), dtype=np.float32)
    thermal[1, 2, 0] = 193.1
    out = PairThermalCalibration(dx=-1, dy=0, fill_value=93.1)(
        {'img2': thermal.copy()})
    assert np.isclose(out['img2'][1, 1, 0], 193.1)
    assert np.isclose(out['img2'][0, 4, 0], 93.1)


def run_all():
    test_identity_and_protection()
    test_negative_x_moves_dot_left()
    test_y_sign_and_rgb_box_invariance()
    test_shift_then_flip_equivalence()
    test_float_fill_and_normalization_nonborder()
    print('PASS test_pair_thermal_calibration')


if __name__ == '__main__':
    run_all()
