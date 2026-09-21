# Phase 3F — Train-only Calibration and P2 Residual Alignment

## Status

This is one 3-epoch seed-0 Lite-P2-P6 screening proxy, not a formal 12-epoch
paper result. SPDI, seed 1/2, GlobalShift training, and a second alignment
variant were not run. The server was left running.

The estimator was designed after observing the historical validation-side
dx=-1, dy=0 result. It is therefore corroborative evidence, not pristine
prospective validation.

## Train-only calibration

All 4,900 train pairs were used. No validation images, annotations, boxes,
predictions, or AP values were read.

Estimator: grayscale RGB/T, Sobel gradient magnitude, common 2-pixel border
crop, zero-mean normalized cross-correlation. The 25 candidates were integer
dx,dy in [-2,2]. The winner by mean ZNCC was dx=-2, dy=+2:

- winner mean ZNCC: 0.0734376082
- second best (-2,+1): 0.0725087828
- margin: 0.0009288254
- bootstrap seed 0, 500 repetitions: winner frequency 1.0
- TRAIN_CALIBRATION: STABLE
- relation to historical (-1,0): INCONSISTENT

All 25 scores are in experiments/phase3f_train_calibration.json and
experiments/phase3f_train_candidate_scores.csv. The winner was used as-is.

Frozen-checkpoint inference-only diagnostic:

| setting | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calibration off | 57.15 | 41.58 | 3.29 | 42.91 | 32.22 | 31.08 | 49.24 | 25.40 |
| train-derived (-2,+2) on | 56.91 | 31.65 | 0.36 | 32.85 | 3.49 | 19.62 | 40.32 | 22.14 |

This does not reproduce the historical shift on a frozen proxy checkpoint.

## Implementation and audit

Added PairThermalCalibration: exact integer thermal-only translation after
Resize and before RandomFlip in train/validation/test. Added P2MicroAlignment:
P2 thermal feature only, bounded to +/-1 input pixel, feature stride 4, and
grid_sample with bilinear interpolation, zero padding, align_corners=False.
GlobalShift is off and SPDI is disabled.

The semantic warm-start source hash matched the Phase 3B proxy. The only
missing keys were p2_micro_alignment.*; unexpected keys and shape mismatches
were empty. Protocol audit: PASS.

Both unit tests passed:

- tests/test_pair_thermal_calibration.py
- tests/test_phase3f_p2_micro_alignment.py

The full-resolution batch-8, 20-iteration smoke passed with finite loss,
backward, and optimizer step.

## Paired proxy

The paired control is the existing Phase 3B Lite-P2 proxy, reused without
retraining.

| run | epoch | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lite-P2 control | 3 | 57.15 | 41.58 | 3.29 | 42.91 | 32.22 | 31.08 | 49.24 | 25.40 |
| CGMA | 1 | 56.22 | 42.09 | 3.18 | 44.16 | 7.72 | 35.39 | 48.49 | 24.41 |
| CGMA | 2 | 56.90 | 42.19 | 3.34 | 44.46 | 12.46 | 32.31 | 51.26 | 19.91 |
| CGMA | 3 | 57.76 | 44.40 | 4.31 | 46.27 | 11.88 | 33.37 | 51.98 | 23.20 |

Best mAP50 epoch: 3. Best minus control in percentage points:

- mAP25 +0.61; mAP50 +2.82; mAP75 +1.02
- tiny +3.36; tiny1 -20.34; tiny2 +2.29; tiny3 +2.74; small -2.20

By the preregistered screening rule, COMBINED_CGMA is STRONG GO for this
proxy only. This is not a stable paper effect: tiny1 and small are volatile
and the subgroup trade-off is substantial.

## Offset dynamics

Values are input-image pixels. The branch receives a finite offset gradient
at iteration 1; reduce-branch gradients appear from iteration 2. At epoch 3,
mean absolute dx/dy were 0.165/0.130 and maximum absolute dx/dy were
0.534/0.458. Saturation ratio was zero. The learned branch was nonzero, not
unused.

## Residual diagnostic

The checkpoint was trained with residual alignment on, so this is diagnostic
rather than a strict retraining ablation.

| setting | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calibration on, residual on | 57.76 | 44.40 | 4.31 | 46.27 | 11.88 | 33.37 | 51.98 | 23.20 |
| calibration on, residual off | 57.86 | 44.73 | 4.59 | 46.62 | 12.39 | 33.66 | 52.36 | 23.42 |
| residual on minus off | -0.10 | -0.33 | -0.28 | -0.35 | -0.51 | -0.29 | -0.38 | -0.22 |

Optional calibration-off, residual-on inference gave mAP50 30.67 versus 44.40
with calibration on; this is also diagnostic because the checkpoint was
trained with calibration on.

Final decisions:

- TRAIN_CALIBRATION = STABLE
- COMBINED_CGMA = STRONG GO for this 3-epoch screening proxy
- RESIDUAL_ALIGNMENT_SIGNAL = NO
- RESIDUAL_UNUSED = NO

The evidence supports a calibration-plus-Lite-P2 proxy observation, but does
not support claiming learned local residual alignment is effective.

## Resources

Proxy wall time was approximately 4,772 seconds (79.5 minutes). Peak allocated
was 20,732,361,216 bytes; peak reserved was 21,701,328,896 bytes; peak driver
usage was 22,173 MiB. Checkpoints remain outside Git.

## Phase 3G status update

Phase 3G freezes Phase 3F. CGMA_ALIGNMENT_CLAIM = NOT ESTABLISHED: the train-only ZNCC winner was the search-boundary corner (-2,+2), frozen calibration degraded mAP50 strongly, and learned P2 residual alignment had no positive diagnostic signal. The 44.40 proxy remains a retraining observation under train-derived fixed translation plus residual branch, not a validated geometric correction.
