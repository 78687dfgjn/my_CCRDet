# Phase 3C - Full Lite-P2 Seed0 Validation

Status: completed. Only the formal Lite-P2-P6 seed=0 run was started. SPDI remained frozen and disabled; no seed=1/2, GlobalShift training, or other new method was started.

## Initialization and protocol

- Config: configs_local/exp_full_lite_p2p6_seed0.py
- Seed 0, deterministic True; initialization: /root/CCRDet/pretrain_weights/resnet50-2stream.pth
- Pretrained SHA256: afa54b9ff80a52402f0489439ccba00f3998b2e49ee1aaffdf2bf78ac5c39639
- load_from=None, resume_from=None; no baseline checkpoint or semantic warm-start
- FPN start_level=0, num_outs=5; detection levels P2-P6; strides [4,8,16,32,64]
- Fusion: Fusion_CAT, Fusion, Fusion, Fusion, Fusion_CAT
- P3/P4/P5 attention: full; GlobalShift off; SPDI disabled; P2 fresh initialized
- Resolved protocol diff: experiments/phase3c_resolved_config_diff.txt

## 5-iteration preflight

PASS. Batch 8, input 512x640, AMP off, gradient accumulation off. Forward, loss, backward, optimizer step completed with finite losses and no OOM/NaN/Inf.

- peak allocated: 19867923456 bytes
- peak reserved: 21533556736 bytes
- peak driver-used: 21621 MiB

## Full training

All 12 epochs completed with validation after every epoch. Wall time: 18922.6 s (5:15:22). Maximum logged training memory: 19249 MiB. Independent driver sampling peak: 20779 MiB; maximum GPU utilization: 100%.

| epoch | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38.81 | 28.17 | 2.06 | 28.64 | 16.75 | 29.96 | 30.88 | 24.01 | 1.4935 |
| 2 | 50.25 | 35.93 | 2.26 | 37.31 | 29.67 | 31.71 | 41.48 | 23.44 | 1.2387 |
| 3 | 52.90 | 37.97 | 2.43 | 39.61 | 20.13 | 28.16 | 44.81 | 25.27 | 1.1899 |
| 4 | 55.85 | 38.54 | 2.67 | 39.75 | 36.88 | 30.64 | 45.82 | 29.55 | 1.1596 |
| 5 | 56.12 | 40.33 | 4.13 | 42.59 | 39.57 | 34.98 | 48.85 | 17.61 | 1.1380 |
| 6 | 56.66 | 42.89 | 4.36 | 45.04 | 42.80 | 31.67 | 51.64 | 20.48 | 1.1174 |
| 7 | 56.54 | 39.18 | 1.87 | 40.27 | 20.41 | 29.19 | 46.75 | 29.70 | 1.1012 |
| 8 | 58.15 | 41.76 | 2.66 | 42.83 | 26.03 | 35.10 | 47.71 | 31.08 | 1.0855 |
| 9 | 59.58 | 43.14 | 3.62 | 44.80 | 21.81 | 32.42 | 51.01 | 26.17 | 1.0099 |
| 10 | 59.87 | 43.21 | 3.79 | 44.90 | 27.98 | 31.54 | 51.72 | 25.16 | 0.9816 |
| 11 | 58.67 | 42.56 | 3.37 | 44.38 | 25.28 | 32.20 | 50.89 | 24.86 | 0.9680 |
| 12 | 58.96 | 43.10 | 4.01 | 44.81 | 19.55 | 32.28 | 51.10 | 26.20 | 0.9492 |

Best mAP50 epoch: 10; checkpoint: /root/CCRDet/work_dirs/phase3c_full_lite_p2p6_seed0/epoch_10.pth.

## Paired comparison with formal CCRDet seed0

Baseline values are from the existing experiments/baseline_seed_0 artifact and were not retrained in this phase.

| metric | baseline seed0 | Lite-P2 best | delta |
|---|---:|---:|---:|
| mAP25 | 57.12 | 59.87 | +2.75 |
| mAP50 | 40.32 | 43.21 | +2.89 |
| mAP75 | 3.49 | 3.79 | +0.30 |
| tiny | 41.94 | 44.90 | +2.96 |
| tiny1 | 13.44 | 27.98 | +14.54 |
| tiny2 | 29.81 | 31.54 | +1.73 |
| tiny3 | 48.18 | 51.72 | +3.54 |
| small | 26.23 | 25.16 | -1.07 |

Epoch12: mAP25=58.96, mAP50=43.10, mAP75=4.01, tiny=44.81, tiny1=19.55, tiny2=32.28, tiny3=51.10, small=26.20.

## Inference-only GlobalShift diagnostic

Using the Lite-P2 best checkpoint, fixed dx=-1, dy=0 was evaluated without retraining.

| metric | dx=0 | dx=-1 | delta |
|---|---:|---:|---:|
| mAP25 | 59.87 | 59.71 | -0.16 |
| mAP50 | 43.21 | 45.96 | +2.75 |
| mAP75 | 3.79 | 4.77 | +0.98 |
| tiny | 44.90 | 47.60 | +2.70 |
| tiny1 | 27.98 | 30.27 | +2.29 |
| tiny2 | 31.54 | 34.68 | +3.14 |
| tiny3 | 51.72 | 53.35 | +1.63 |
| small | 25.16 | 25.39 | +0.23 |

This is an inference-only observation, not a trained GlobalShift result or a formal innovation.

## Final status

- LITE_P2 = CONFIRMED USEFUL COMPONENT
- TINY1 BENEFIT = YES
- SMALL TRADE-OFF = YES
- HIGH-IOU LOCALIZATION BENEFIT = YES

These labels summarize one seed, not a multi-seed statistical claim. No automatic SPDI, second seed, refinement, loss, assigner, or augmentation work was started.
