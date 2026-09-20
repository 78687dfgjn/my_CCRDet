# RGBTDronePerson tiny evaluation protocol

This audit follows the released repository call stack and does not modify the
detector or evaluator.

## Call stack

`tools/test.py` builds the test dataset in `test_mode=True`, runs
`mmdet.apis.single_gpu_test`, and then calls `dataset.evaluate(results,
metric='bbox')`. The detector path is `GFLAF.simple_test` -> `GFLHead` test
decoding -> `multiclass_nms`.

The model-side test configuration is:

| setting | value |
|---|---:|
| `nms_pre` | 1000 |
| score threshold | 0.05, strict `scores > score_thr` |
| NMS | class-wise NMS, IoU 0.30 |
| final detections per image | 100 |

The dataset path is then `DronePerson.evaluate` -> `COCO.loadRes` -> the
repository's `mmdet.datasets.evaluation.coco.COCOeval`. `DronePerson.evaluate`
sets `Params.EVAL_STRANDARD='tiny'`, `ignore_uncertain=True`,
`use_iod_for_ignore=True`, `iouThrs=[0.25, 0.50, 0.75]`, and explicitly sets
`cocoEval.params.maxDets = [100, 300, 1000]`.

## maxDets=200 versus maxDets=1000

`Params.setDetParams()` initializes the tiny protocol with `maxDets=[200]`.
That value is not the final released evaluation setting: `DronePerson.evaluate`
overwrites it with the `proposal_nums` default `[100,300,1000]`. The actual
official `tools/test.py --eval bbox` output prints `maxDets=1000` and uses
1000 as the per-image evaluator truncation limit.

The historical analysis scripts that hard-code `MAX_DETS=200` are therefore
offline analysis protocols, not the released `tools/test.py` evaluation
protocol.

## Matching and ignore behavior

At each IoU threshold, detections are sorted by descending score using a
stable sort and truncated to `maxDets[-1]`. Matching is category-aware. A
non-ignored GT is matched at or above the current IoU threshold; ignored GTs
are sorted after valid GTs. The evaluator additionally marks detections
overlapping ignored regions through its repository-specific IOD path. A
detection outside the selected area range is ignored for that area range.

The tiny area ranges in source are:

| label | source range |
|---|---:|
| all | `[1, 100000^2]` |
| tiny | `[1, 20^2]` |
| tiny1 | `[1, 8^2]` |
| tiny2 | `[8^2, 12^2]` |
| tiny3 | `[12^2, 20^2]` |
| small | `[20^2, 32^2]` |
| reasonable | `[32^2, 100000^2]` |

The comparisons in `evaluateImg` are inclusive (`< lower` or `> upper` means
outside), so exact boundary values belong to both adjacent buckets. The
released model-side NMS happens before COCOeval; COCOeval itself does not run
another NMS pass.

## Reproduction check

Using `weight_rgbt.pth` and the path-only local config reproduced:

| metric | value |
|---|---:|
| mAP25 all | 0.5847 |
| mAP50 all | 0.4361 |
| mAP75 all | 0.0462 |
| mAP50 tiny | 0.4514 |
| mAP50 tiny1 | 0.2521 |

The run used 1225 test images because `tools/test.py` sets `test_mode=True`.
The direct default dataset constructor filters nine uncertain-only images;
that distinction is documented and not silently mixed into this protocol.
