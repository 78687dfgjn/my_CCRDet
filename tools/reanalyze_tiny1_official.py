"""Re-analyse tiny1 instances using the released DronePerson evaluator.

This is an analysis-only script.  It consumes detector results produced by
``tools/test.py`` and never changes detector or evaluator code.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config

from mmdet.datasets import build_dataset
from mmdet.datasets.evaluation.coco.cocoeval import COCOeval, Params


def xywh_iou(a, b):
    ax1, ay1, aw, ah = [float(x) for x in a[:4]]
    bx1, by1, bw, bh = [float(x) for x in b[:4]]
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def eval_img_map(coco_eval, area_label, max_det):
    area_idx = coco_eval.params.areaRngLbl.index(area_label)
    n_img = len(coco_eval.params.imgIds)
    n_cat = len(coco_eval.params.catIds)
    out = {}
    for cat_idx, cat_id in enumerate(coco_eval.params.catIds):
        for img_idx, img_id in enumerate(coco_eval.params.imgIds):
            flat = cat_idx * len(coco_eval.params.areaRngLbl) * n_img
            flat += area_idx * n_img + img_idx
            item = coco_eval.evalImgs[flat]
            if item is not None and item['maxDet'] == max_det:
                out[(img_id, cat_id)] = item
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.results)
    if len(results) != len(dataset):
        raise RuntimeError(f'results={len(results)} dataset={len(dataset)}')

    result_files, tmp_dir = dataset.format_results(results, jsonfile_prefix=str(out_dir / 'official'))
    predictions = mmcv.load(result_files['bbox'])
    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(predictions)

    Params.EVAL_STRANDARD = 'tiny'
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox',
                         ignore_uncertain=True, use_iod_for_ignore=True)
    evaluator.params.catIds = dataset.cat_ids
    evaluator.params.imgIds = dataset.img_ids
    evaluator.params.maxDets = [100, 300, 1000]
    evaluator.params.iouThrs = np.array([0.25, 0.50, 0.75])
    evaluator.evaluate()
    per_image = eval_img_map(evaluator, 'tiny1', 1000)

    cat_names = {cat_id: coco_gt.cats[cat_id]['name'] for cat_id in dataset.cat_ids}
    records = []
    counts = {k: 0 for k in ['A', 'B', 'C', 'D']}
    by_category = {}
    for img_id in dataset.img_ids:
        for cat_id in dataset.cat_ids:
            item = per_image.get((img_id, cat_id))
            if item is None:
                continue
            gt_by_id = {ann['id']: ann for ann in coco_gt.loadAnns(item['gtIds'])}
            dt_by_id = {ann['id']: ann for ann in coco_dt.loadAnns(item['dtIds'])}
            # The official evaluator puts ignored/out-of-range GT after valid GT.
            for gt_pos, (gt_id, ignored) in enumerate(zip(item['gtIds'], item['gtIgnore'])):
                if int(ignored):
                    continue
                gt = gt_by_id[gt_id]
                candidates = []
                for dt_id, dt in dt_by_id.items():
                    if dt.get('category_id') != cat_id:
                        continue
                    iou = xywh_iou(gt['bbox'], dt['bbox'])
                    candidates.append((iou, float(dt['score']), dt_id, dt))
                candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
                best_iou, best_score, best_dt_id, best_dt = (
                    candidates[0] if candidates else (0.0, float('nan'), None, None))
                highest_score = max((x[1] for x in candidates), default=float('nan'))
                formal_dt_id = int(item['gtMatches'][1, gt_pos])
                formal_tp = formal_dt_id > 0
                if formal_tp:
                    failure_type = 'D'
                elif best_iou >= 0.5:
                    failure_type = 'C'
                elif best_iou >= 0.25:
                    failure_type = 'B'
                else:
                    failure_type = 'A'
                counts[failure_type] += 1
                by_category.setdefault(cat_names[cat_id], {k: 0 for k in counts})
                by_category[cat_names[cat_id]][failure_type] += 1
                gt_w, gt_h = float(gt['bbox'][2]), float(gt['bbox'][3])
                if best_dt is None:
                    pred_bbox = [None, None, None, None]
                    pred_w = pred_h = ratio = float('nan')
                else:
                    pred_bbox = [float(x) for x in best_dt['bbox']]
                    pred_w, pred_h = pred_bbox[2], pred_bbox[3]
                    ratio = math.sqrt(max(pred_w * pred_h, 0.0) /
                                      max(gt_w * gt_h, 1e-12))
                records.append({
                    'image_id': int(img_id),
                    'ann_id': int(gt_id),
                    'category_id': int(cat_id),
                    'category': cat_names[cat_id],
                    'bbox_xywh': [float(x) for x in gt['bbox']],
                    'area': float(gt['area']),
                    'gt_width': gt_w,
                    'gt_height': gt_h,
                    'matched_status': 'TP' if formal_tp else 'FN',
                    'failure_type': failure_type,
                    'formal_tp_prediction_id': formal_dt_id if formal_tp else None,
                    'highest_prediction_score': highest_score,
                    'best_iou': float(best_iou),
                    'best_candidate_score': best_score,
                    'iou_ge_025_candidate': bool(best_iou >= 0.25),
                    'iou_ge_050_candidate': bool(best_iou >= 0.50),
                    'best_candidate_prediction_id': int(best_dt_id) if best_dt_id else None,
                    'best_candidate_bbox_xywh': pred_bbox,
                    'pred_width_over_gt_width': pred_w / gt_w if gt_w else float('nan'),
                    'pred_height_over_gt_height': pred_h / gt_h if gt_h else float('nan'),
                    'sqrt_pred_area_over_gt_area': ratio,
                    'prediction_fpn_level': 'UNKNOWN_POST_NMS',
                })

    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v

    records = [clean(x) for x in records]
    summary = {
        'protocol': {
            'area_label': 'tiny1',
            'area_range': [1, 64],
            'iou_thresholds': [0.25, 0.5, 0.75],
            'matching_iou': 0.5,
            'max_dets': [100, 300, 1000],
            'matching_max_det': 1000,
            'categories': cat_names,
            'ignore_uncertain': True,
            'use_iod_for_ignore': True,
        },
        'counts': counts,
        'by_category': by_category,
        'total_tiny1_gt': len(records),
        'candidate_fpn_level': 'Not recoverable from post-NMS result tensors.',
    }
    (out_dir / 'tiny1_instances.json').write_text(json.dumps(records, indent=2, allow_nan=False))
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    fields = list(records[0].keys()) if records else []
    with (out_dir / 'tiny1_instances.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (out_dir / 'protocol.json').write_text(json.dumps(summary['protocol'], indent=2))
    if tmp_dir is not None:
        tmp_dir.cleanup()
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
