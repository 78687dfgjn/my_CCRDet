# Paired CCRDet-P2-P7 training plan (not run)

This file records a configuration-only candidate for the already completed
CCRDet seed-0 baseline. No training was started in Phase 2C.

## Configs

- baseline: `configs_local/gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py`
- candidate: `configs_local/gfl_ccrdet_fpn_1x_rgbtdroneperson_p2_p7_seed0.py`
- paired checkpoint reference: CCRDet seed 0, reported mAP50 = 40.32

## Resolved semantic diff

| field | baseline | candidate |
|---|---|---|
| FPN `start_level` | 1 | 0 |
| FPN `num_outs` | 5 | 6 |
| FPN output strides | 8,16,32,64,128 | 4,8,16,32,64,128 |
| bbox head anchor strides | 8,16,32,64,128 | 4,8,16,32,64,128 |
| seed | 0 | 0 |
| backbone | unchanged | unchanged |
| CRG/IPM/LSK | unchanged | unchanged |
| fusion modules | unchanged by config | unchanged by config |
| head/loss/optimizer/schedule | unchanged | unchanged |
| augmentation/input/NMS | unchanged | unchanged |

## Compatibility gate

The current `mmdet/models/detectors/afdet.py` constructs exactly five fusion
modules and indexes one for every FPN output. A six-output candidate therefore
cannot be trained safely from configuration alone: it would reach an index
mismatch in the detector forward path. This plan intentionally does not patch
that detector because Phase 2C forbids new model implementation. A future P2
experiment is **not ready to train** until a separately reviewed, numerical-
behavior-preserving multi-level compatibility change is prepared and tested.
