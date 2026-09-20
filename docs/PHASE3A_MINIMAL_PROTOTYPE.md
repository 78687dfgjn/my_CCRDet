# Phase 3A — Minimal Prototype Integration

## Scope

This phase implements only two reversible research prototypes: `GlobalThermalShift` and a five-level P2–P6 candidate. No TDS, JDFF, reliability, local alignment, box refinement, new loss, or new augmentation was added. No seed=1/2 training was run.

All metrics below are screening evidence, not formal paper results.

## 1. Files changed

- `mmdet/models/utils/global_shift.py`
- `mmdet/models/utils/__init__.py`
- `mmdet/models/detectors/afdet.py`
- `configs_local/exp_baseline_seed0.py`
- `configs_local/exp_global_fixed_seed0.py`
- `configs_local/exp_p2p6_seed0.py`
- `configs_local/exp_global_p2p6_seed0.py`
- `tools/make_phase3a_warmstart.py`
- `tests/test_global_shift.py`

Generated records are in `experiments/phase3a_*` and `experiments/phase3a_screening_metrics.json`.

## 2. GlobalThermalShift

The module is inserted after paired image transforms and before the two-stream backbone. It affects only the thermal tensor.

- `off`: returns the original tensor directly.
- `fixed`: applies configured `dx,dy` in input pixels.
- `learnable`: uses two scalar parameters with `max_shift * tanh(raw_shift)`.
- `dx < 0` means thermal content moves left.
- `grid_sample` uses zero padding and differentiable bilinear sampling.

Unit checks all passed:

1. `off` is identity and returns equal values.
2. fixed `(0,0)` is exact identity.
3. synthetic dot with `dx=-1` moves from x=3 to x=2.
4. gradients reach both learnable shift parameters.

## 3. Fusion refactor and compatibility

`GFLAF` now accepts `fusion_types` from config. The default remains:

```text
P3/P4/P5 -> Fusion
P6/P7    -> Fusion_CAT
```

The P2–P6 prototype uses:

```text
P2/P3/P4 -> Fusion
P5/P6    -> Fusion_CAT
```

With the official checkpoint and fixed random input, the official config and `exp_baseline_seed0.py` produced:

```text
strict state_dict load: 830/830 keys
feature max abs diff:    0.0
```

Thus the mode-off baseline configuration is numerically compatible with the current official configuration.

## 4. P2–P6 shapes

For a smoke input of `[1,3,128,160]`, the candidate returned five tensors:

```text
P2: [1,256,32,40]
P3: [1,256,16,20]
P4: [1,256,8,10]
P5: [1,256,4,5]
P6: [1,256,2,3]
```

The configured detection strides are `[4,8,16,32,64]`; P7 is not retained.

## 5. Warm-start provenance

Source: `/root/CCRDet/work_dirs/baseline_seed_0/epoch_12.pth`.

The candidate has 834 state keys versus 830 in the source. Shape-compatible filtering produced:

| item | count |
|---|---:|
| compatible loaded keys | 824 |
| missing/new candidate keys | 4 |
| same-name shape mismatches | 6 |
| unexpected source keys | 0 |
| newly initialized keys total | 10 |

The six mismatches are the RGB/T FPN lateral convolution weights whose input channel semantics change when `start_level` moves from 1 to 0. They were intentionally reinitialized. Full details are in `experiments/phase3a_warmstart_keys.json`.

## 6. Screening metrics

All values are AP percentage points. A is the existing seed-0 baseline checkpoint; B uses the same checkpoint with fixed `dx=-1,dy=0` inference.

| experiment | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A baseline seed0 | 57.12 | 40.32 | 3.49 | 41.94 | 13.44 | 29.81 | 48.18 | 26.23 |
| B fixed shift, seed0 checkpoint | 57.01 | 42.17 | 4.74 | 43.85 | 21.53 | 32.58 | 49.26 | 26.24 |
| B fixed shift, official checkpoint | 58.28 | 44.07 | 5.53 | 45.34 | 18.68 | 35.68 | 50.36 | 30.45 |

Paired A→B seed0 deltas are: mAP50 `+1.85 pp`, tiny `+1.91 pp`, tiny1 `+8.09 pp`, tiny2 `+2.77 pp`, tiny3 `+1.08 pp`, and small `+0.01 pp`. mAP25 changes by `-0.11 pp`; mAP75 changes by `+1.25 pp`.

## 7. C/D proxy status

C and D were both run with the same seed=0, deterministic setting, batch size 8, 640×512 input, 3-epoch limit, and proxy learning rate 0.001. Both failed before the first training iteration with CUDA OOM in the first P2 `Fusion` attention multiplication.

- C requested 12.50 GiB while 13.36 GiB was allocated and 13.38 GiB reserved; 21.67 GiB total.
- D requested 12.50 GiB while 13.39 GiB was allocated and 13.41 GiB reserved; 21.67 GiB total.

No C/D AP is reported because no training iteration completed. Batch size, AMP, gradient accumulation, attention, and image resolution were not changed to force a result.

## 8. Decisions

- Global fixed shift: `GO` for a separately gated formal seed=0 12-epoch run, if explicitly authorized. The current evidence is still a prototype result, not a paper claim.
- P2–P6: `NO-GO` under the current fusion mapping because the required protocol OOMs before training.
- Global fixed + P2–P6: `NO-GO` for the same reason.

The P2 rejection is an engineering/protocol decision for this implementation, not evidence that every possible P2 design is ineffective. Reviving P2 would require a separately approved memory-safe fusion design, which was not implemented in Phase 3A.

## 9. Unresolved issues

- GlobalShift has not yet been evaluated with a formal 12-epoch seed=0 training run.
- The P2 candidate has no valid proxy AP because of the prescribed-protocol OOM.
- Learnable GlobalShift was unit-tested only; it was not trained.
