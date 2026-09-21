"""Train-only ATSS/anchor geometry audit for Phase 3I."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmdet.core.anchor import AnchorGenerator


TARGET_W, TARGET_H = 640, 512
STRIDES = [4, 8, 16, 32, 64]
GROUPS = {
    'tiny': (1.0, 400.0),
    'tiny1': (1.0, 64.0),
    'tiny2': (64.0, 144.0),
    'tiny3': (144.0, 400.0),
    'small': (400.0, 1024.0),
}


def load_annotations(path):
    raw = json.loads(Path(path).read_text())
    images = {int(x['id']): x for x in raw['images']}
    allowed = {int(x['id']) for x in raw.get('categories', [])
               if x.get('name') in ('person', 'rider', 'crowd')}
    by_image = defaultdict(list)
    for ann in raw['annotations']:
        x, y, w, h = [float(v) for v in ann['bbox']]
        info = images[int(ann['image_id'])]
        inter_w = max(0.0, min(x + w, float(info['width'])) - max(x, 0.0))
        inter_h = max(0.0, min(y + h, float(info['height'])) - max(y, 0.0))
        if ann.get('ignore', False) or ann.get('iscrowd', False):
            continue
        if int(ann['category_id']) not in allowed:
            continue
        if float(ann.get('area', w * h)) <= 0 or w < 1 or h < 1:
            continue
        if inter_w * inter_h <= 0:
            continue
        by_image[int(ann['image_id'])].append({
            'bbox': [x, y, w, h],
            'area': float(ann.get('area', w * h)),
            'category_id': int(ann['category_id']),
        })
    return raw, images, by_image


def resized_boxes(image, anns):
    scale = min(TARGET_W / float(image['width']),
                TARGET_H / float(image['height']))
    out = []
    for ann in anns:
        x, y, w, h = ann['bbox']
        out.append({
            'xyxy': np.asarray([x, y, x + w, y + h], dtype=np.float32) * scale,
            'area_eval': ann['area'],
            'width_raw': w,
            'height_raw': h,
            'area_raw': w * h,
            'category_id': ann['category_id'],
        })
    resized_w = int(round(float(image['width']) * scale))
    resized_h = int(round(float(image['height']) * scale))
    padded_w = int(np.ceil(resized_w / 32.0) * 32)
    padded_h = int(np.ceil(resized_h / 32.0) * 32)
    return out, (padded_h, padded_w), scale


def make_generator(p2_side):
    base_sizes = [float(p2_side) / 8.0, 8.0, 16.0, 32.0, 64.0]
    return AnchorGenerator(
        strides=STRIDES,
        ratios=[1.0],
        base_sizes=base_sizes,
        octave_base_scale=8,
        scales_per_octave=1)


def iou_one(anchors, box):
    x1 = np.maximum(anchors[:, 0], box[0])
    y1 = np.maximum(anchors[:, 1], box[1])
    x2 = np.minimum(anchors[:, 2], box[2])
    y2 = np.minimum(anchors[:, 3], box[3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    aa = np.maximum(anchors[:, 2] - anchors[:, 0], 0) * np.maximum(anchors[:, 3] - anchors[:, 1], 0)
    ba = max(float(box[2] - box[0]), 0) * max(float(box[3] - box[1]), 0)
    return inter / np.maximum(aa + ba - inter, 1e-12)


def group_name(area):
    names = []
    for name, (lo, hi) in GROUPS.items():
        if area >= lo and area <= hi:
            names.append(name)
    return names


def assignment_for_image(anchors_by_level, boxes, topk=9):
    flat = np.concatenate(anchors_by_level, axis=0)
    centers = (flat[:, :2] + flat[:, 2:]) / 2.0
    level_ranges = []
    start = 0
    for level, anchors in enumerate(anchors_by_level):
        end = start + len(anchors)
        level_ranges.append((start, end))
        start = end
    n_gt = len(boxes)
    if not n_gt:
        return []
    candidate_ids = []
    candidate_ious = []
    candidate_dists = []
    for b in boxes:
        box = b['xyxy']
        gc = np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)
        ids_per_level, ious_per_level, dists_per_level = [], [], []
        for start, end in level_ranges:
            d = np.sqrt(((centers[start:end] - gc) ** 2).sum(axis=1))
            k = min(topk, len(d))
            idx = torch.from_numpy(d).topk(k, largest=False).indices.numpy()
            idx = idx + start
            ids_per_level.append(idx)
            ious_per_level.append(iou_one(flat[idx], box))
            dists_per_level.append(d[idx - start])
        candidate_ids.append(np.concatenate(ids_per_level))
        candidate_ious.append(np.concatenate(ious_per_level))
        candidate_dists.append(np.concatenate(dists_per_level))
    candidate_ids = np.stack(candidate_ids, axis=1)       # [45, G]
    candidate_ious = np.stack(candidate_ious, axis=1)     # [45, G]
    candidate_dists = np.stack(candidate_dists, axis=1)   # [45, G]
    thresholds = candidate_ious.mean(axis=0) + candidate_ious.std(axis=0, ddof=1)
    is_in = np.zeros_like(candidate_ious, dtype=bool)
    for j, b in enumerate(boxes):
        box = b['xyxy']
        c = centers[candidate_ids[:, j]]
        is_in[:, j] = ((c[:, 0] - box[0] > 0.01) &
                       (c[:, 1] - box[1] > 0.01) &
                       (box[2] - c[:, 0] > 0.01) &
                       (box[3] - c[:, 1] > 0.01))
    is_pos = (candidate_ious >= thresholds[None, :]) & is_in

    best_by_anchor = {}
    for row, gt in zip(*np.nonzero(is_pos)):
        anchor_id = int(candidate_ids[row, gt])
        score = float(candidate_ious[row, gt])
        old = best_by_anchor.get(anchor_id)
        if old is None or score > old[1]:
            best_by_anchor[anchor_id] = (gt, score)
    final_by_gt = defaultdict(list)
    for anchor_id, (gt, score) in best_by_anchor.items():
        final_by_gt[gt].append((anchor_id, score))

    records = []
    level_starts = np.cumsum([0] + [len(x) for x in anchors_by_level])
    for j, b in enumerate(boxes):
        cand_level_iou = []
        cand_level_dist = []
        for level in range(len(anchors_by_level)):
            a, z = level_starts[level], level_starts[level + 1]
            mask = (candidate_ids[:, j] >= a) & (candidate_ids[:, j] < z)
            cand_level_iou.append(float(candidate_ious[mask, j].max()))
            cand_level_dist.append(float(candidate_dists[mask, j].mean()))
        positives = final_by_gt.get(j, [])
        pos_counts = [0] * len(anchors_by_level)
        pos_ious = []
        pos_dists = []
        for anchor_id, score in positives:
            level = int(np.searchsorted(level_starts[1:], anchor_id, side='right'))
            pos_counts[level] += 1
            pos_ious.append(score)
            pos_dists.append(float(np.linalg.norm(centers[anchor_id] - ((b['xyxy'][:2] + b['xyxy'][2:]) / 2.0))))
        records.append({
            'area': float(b['area_eval']),
            'groups': group_name(float(b['area_eval'])),
            'width': float(b['width_raw']),
            'height': float(b['height_raw']),
            'sqrt_area': float(np.sqrt(max(b['area_raw'], 0))),
            'threshold': float(thresholds[j]),
            'positive_count': int(len(positives)),
            'positive_count_by_level': pos_counts,
            'best_candidate_iou_by_level': cand_level_iou,
            'best_candidate_iou': float(max(cand_level_iou)),
            'best_positive_iou': float(max(pos_ious) if pos_ious else 0.0),
            'candidate_center_distance_mean_by_level': cand_level_dist,
            'positive_center_distance_mean': float(np.mean(pos_dists) if pos_dists else 0.0),
            'zero_positive': not bool(positives),
        })
    return records


def summarize(records):
    levels = ['P2', 'P3', 'P4', 'P5', 'P6']
    result = {}
    for name in ['all'] + list(GROUPS):
        selected = records if name == 'all' else [r for r in records if name in r['groups']]
        if not selected:
            result[name] = {'count': 0}
            continue
        pos = np.asarray([r['positive_count'] for r in selected], dtype=np.float64)
        best_cand = np.asarray([r['best_candidate_iou'] for r in selected], dtype=np.float64)
        best_pos = np.asarray([r['best_positive_iou'] for r in selected], dtype=np.float64)
        thresholds = np.asarray([r['threshold'] for r in selected], dtype=np.float64)
        counts = np.asarray([r['positive_count_by_level'] for r in selected], dtype=np.float64)
        total_pos = max(float(counts.sum()), 1.0)
        result[name] = {
            'count': int(len(selected)),
            'mean_positive_count': float(pos.mean()),
            'median_positive_count': float(np.median(pos)),
            'zero_positive_rate': float(np.mean(pos == 0)),
            'positive_fraction_by_level': {level: float(counts[:, i].sum() / total_pos) for i, level in enumerate(levels)},
            'median_best_candidate_iou': float(np.median(best_cand)),
            'median_atss_threshold': float(np.median(thresholds)),
            'median_best_positive_iou': float(np.median(best_pos)),
            'mean_best_candidate_iou': float(best_cand.mean()),
            'mean_best_positive_iou': float(best_pos.mean()),
            'mean_positive_center_distance': float(np.mean([r['positive_center_distance_mean'] for r in selected])),
        }
    return result


def write_geometry(raw, images, by_image):
    vals = []
    for anns in by_image.values():
        for a in anns:
            x, y, w, h = a['bbox']
            vals.append({'width': w, 'height': h, 'sqrt_area': np.sqrt(max(w*h, 0)), 'max_side': max(w,h), 'min_side': min(w,h), 'aspect_ratio': max(w,h)/max(min(w,h),1e-12), 'area': a['area']})
    keys = ['width','height','sqrt_area','max_side','min_side','aspect_ratio','area']
    out = {'data_split':'train','image_count':len(images),'pair_count':len(images),'gt_count':len(vals),'statistics':{}}
    for k in keys:
        x=np.asarray([v[k] for v in vals], dtype=np.float64)
        out['statistics'][k]={('p%d'%p):float(np.percentile(x,p)) for p in (10,25,50,75,90,95,98)}
    out['evaluator_size_bins']={k:{'lower_area':v[0],'upper_area':v[1],'boundary_inclusive':True} for k,v in GROUPS.items()}
    out['evaluator_source']='mmdet/datasets/evaluation/coco/cocoeval.py Params.setDetParams: area filter ignores only < lower or > upper.'
    Path('experiments/phase3i_train_gt_geometry.json').write_text(json.dumps(out,indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--p2-side',type=float,required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--ann-file',default='train_thermal.json')
    args=ap.parse_args()
    raw,images,by_image=load_annotations(args.ann_file)
    write_geometry(raw,images,by_image)
    generator=make_generator(args.p2_side)
    base=[a.numpy() for a in generator.base_anchors]
    records=[]
    image_shapes=defaultdict(int)
    for image_id, anns in by_image.items():
        boxes, padded_shape, scale=resized_boxes(images[image_id], anns)
        image_shapes[str(padded_shape)] += 1
        feat_shapes=[(padded_shape[0]//s,padded_shape[1]//s) for s in STRIDES]
        priors=[x.numpy() for x in generator.grid_priors(feat_shapes,device='cpu')]
        records.extend(assignment_for_image(priors, boxes, topk=9))
    widths=[[float(x[2]-x[0]) for x in level] for level in base]
    heights=[[float(x[3]-x[1]) for x in level] for level in base]
    result={
        'status':'PASS', 'data_split':'train', 'annotation_file':str(args.ann_file), 'image_count':len(images), 'gt_count':len(records),
        'p2_anchor_side_requested':args.p2_side, 'strides':STRIDES, 'ratios':[1.0], 'octave_base_scale':8, 'scales_per_octave':1,
        'actual_base_anchor_widths':widths, 'actual_base_anchor_heights':heights, 'num_base_priors':generator.num_base_priors,
        'feature_shapes_by_padded_image_shape':dict(image_shapes), 'atss':{'topk':9,'threshold':'mean(candidate IoUs)+unbiased std(candidate IoUs)','center_inside_rule':'>0.01','conflict_rule':'highest IoU among positive candidate assignments'},
        'group_statistics':summarize(records),
    }
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
