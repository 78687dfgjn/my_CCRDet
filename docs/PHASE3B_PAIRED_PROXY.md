# Phase 3B Paired Proxy Report

## Scope

This phase corrected the Lite-P2 attention-path variable and ran only the requested
low-cost screening protocol. No full 12-epoch training, seed 1/2 training, GlobalShift
training, or new research module was run.

The paired proxy protocol is:

- seed 0, deterministic mode;
- batch size 8, input resolution 640x512;
- SGD with `lr=0.001`;
- three epochs, validation after every epoch;
- `load_from` from the same released seed-0 epoch-12 checkpoint, without optimizer-state resume;
- AMP and gradient accumulation disabled;
- unchanged dataset, augmentation, loss, NMS, and evaluation protocol.

## 1. Attention behavior correction

`Fusion` now defaults to `attention_mode='full'`. `GFLAF` maps an omitted
`attention_mode` to `full`, so the released baseline behavior is not implicitly
changed by the memory guard. The skip guard remains available only as an explicit
research/debug option.

The Lite-P2 configuration uses:

```text
P2 -> Fusion_CAT
P3 -> Fusion
P4 -> Fusion
P5 -> Fusion
P6 -> Fusion_CAT
```

## 2. Formal-resolution fusion-path verification

Input was `[1, 3, 512, 640]` for each modality.

| semantic level | feature shape | module | attention path |
|---|---:|---|---|
| P2 | `[1,256,128,160]` | Fusion_CAT | n/a |
| P3 | `[1,256,64,80]` | Fusion | full, skip calls = 0 |
| P4 | `[1,256,32,40]` | Fusion | full |
| P5 | `[1,256,16,20]` | Fusion | full |
| P6 | `[1,256,8,10]` | Fusion_CAT | n/a |

The P3 call has `H*W=5120`, which is above the optional guard threshold 4096,
but it still executed the full path because the candidate explicitly uses the
faithful default. The test result is `PASS`; details are in
`experiments/phase3b_fusion_path_check.json`.

## 3. Semantic warm-start verification

The FPN and fusion modules were remapped by semantic level rather than by raw
index:

| source baseline | candidate target | action |
|---|---|---|
| P3 / index 0 | P3 / index 1 | load |
| P4 / index 1 | P4 / index 2 | load |
| P5 / index 2 | P5 / index 3 | load |
| P6 / index 3 | P6 / index 4 | load |
| P7 / index 4 | — | discard |
| — | P2 / index 0 | fresh initialization |

This mapping applies to both RGB/thermal FPN lateral and output convolutions and
to the fusion list. Statistics:

- loaded keys: 824;
- newly initialized keys: 10;
- P2 newly initialized parameter count: 1,443,072;
- discarded keys: 6;
- semantic state-dict sanity check: `PASS`.

The machine-readable records are
`experiments/phase3a_semantic_warmstart_report.json` and
`experiments/phase3a_semantic_warmstart_sanity.json`.

## 4. Full-resolution batch-8 smoke test

The real training data path was used for 20 forward/loss/backward/optimizer-step
iterations. AMP and gradient accumulation were disabled.

| item | result |
|---|---:|
| status | PASS |
| completed iterations | 20 / 20 |
| mean iteration time | 2.216 s |
| peak torch allocated | 19,869,149,184 bytes (18.51 GiB) |
| peak torch reserved | 21,533,556,736 bytes (20.06 GiB) |
| peak driver-used | 21,621 MiB |
| NaN/Inf/OOM | none |

Feature shapes were `[8,256,128,160]`, `[8,256,64,80]`,
`[8,256,32,40]`, `[8,256,16,20]`, and `[8,256,8,10]` for P2 through P6.
The complete record is `experiments/phase3b_batch8_smoke.json`.

The run passes, but the driver-used peak leaves limited headroom on the 22,528 MiB
GPU and should be treated as a resource warning for future longer runs.

## 5. Paired three-epoch proxy metrics

Values below are percentages (the raw evaluator fractions are preserved in the logs).

| model | epoch | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CONTROL P3-P7 | 1 | 56.65 | 38.28 | 2.74 | 40.10 | 18.81 | 27.89 | 46.76 | 24.10 |
| CONTROL P3-P7 | 2 | 55.80 | 38.02 | 2.98 | 39.42 | 19.44 | 26.95 | 46.18 | 25.45 |
| CONTROL P3-P7 | 3 | 56.35 | 40.01 | 3.46 | 41.30 | 13.56 | 29.11 | 47.60 | 28.14 |
| Lite-P2-P6 | 1 | 55.95 | 33.77 | 1.12 | 35.09 | 24.24 | 24.60 | 41.41 | 24.70 |
| Lite-P2-P6 | 2 | 56.54 | 40.01 | 2.65 | 41.61 | 26.12 | 30.08 | 48.12 | 23.61 |
| Lite-P2-P6 | 3 | 57.15 | 41.58 | 3.29 | 42.91 | 32.22 | 31.08 | 49.24 | 25.40 |

Both models select epoch 3 by validation mAP50.

### Best-to-best paired delta

`Lite-P2-P6 - CONTROL`, both at epoch 3:

| metric | delta (percentage points) |
|---|---:|
| mAP25 | +0.80 |
| mAP50 | +1.57 |
| mAP75 | -0.17 |
| tiny | +1.61 |
| tiny1 | +18.66 |
| tiny2 | +1.97 |
| tiny3 | +1.64 |
| small | -2.74 |

The absolute 12-epoch seed-0 baseline number (40.32) is not used as the paired
proxy control; comparing it directly to this three-epoch fine-tuning result would
confound training duration.

## 6. Screening decision

`LITE_P2 = GO` for the next controlled experiment only. The proxy exceeds the
requested +0.8 pp mAP50 criterion and improves tiny1/tiny2 while overall mAP50
does not decrease. The small-group decrease is a material trade-off and the result
is not a paper claim; it must be rechecked under a full, separately controlled
12-epoch training before drawing conclusions.

GlobalShift was not trained or mixed into this decision.
