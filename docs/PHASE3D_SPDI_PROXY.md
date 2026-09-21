# Phase 3D — SPDI Paired Proxy Validation

This is a 3-epoch screening result, not a formal paper result. No seed 1/2, 12-epoch SPDI run, new loss, assigner, augmentation, or SPDI architecture change was performed.

## Protocol and initialization

- Existing Phase 3B control was reused: same seed 0, deterministic mode, epoch-12 baseline checkpoint, SGD lr=0.001, batch 8, 640×512, 3 epochs, data pipeline, loss, and evaluator.
- Initialization audit: PASS. Baseline checkpoint had 9 missing keys, all under `p2_detail_injection.*`; unexpected keys: 0. Semantic warm-start was not used.
- Formal topology remained P3–P7 with strides `[8,16,32,64,128]`, FPN `start_level=1,num_outs=5`, Fusion/Fusion/Fusion/Fusion_CAT/Fusion_CAT, GlobalShift off. `attention_mode=full` is explicit and matches the detector default.
- SPDI parameter count: 200,194 (P2 fusion 131,328; depthwise 2,560; pointwise 65,792; gate 513; alpha 1).

## Metrics

| epoch | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 56.62 | 38.13 | 2.73 | 39.87 | 18.70 | 27.76 | 46.57 | 24.05 | 0.9149 |
| 2 | 55.80 | 38.20 | 3.01 | 39.64 | 19.44 | 27.16 | 46.24 | 25.05 | 0.9074 |
| 3 | 56.39 | 40.17 | 3.50 | 41.33 | 19.24 | 29.35 | 47.68 | 27.95 | 0.8892 |

Paired control best (epoch 3): mAP50 40.01, tiny 41.30, tiny1 13.56, tiny2 29.11, tiny3 47.60, small 28.14.
SPDI best epoch: 3. Deltas vs paired control: mAP50 +0.16, tiny +0.03, tiny1 +5.68, tiny2 +0.24, tiny3 +0.08, small -0.19, mAP75 +0.04.

## Alpha and branch dynamics

- alpha was initialized at 0. It moved to approximately 4.016e-11 after iteration 1 and 9.269e-05 at epoch 3 end.
- Iteration-1 branch gradients were exactly zero, as expected from the zero alpha path; from iteration 2 onward the branch gradients became finite and nonzero. At epoch 3 end: p2_fusion 2.420e-06, depthwise 5.144e-07, pointwise 2.785e-06, gate 9.500e-07; alpha grad -3.430e-04.
- All 10 recorded dynamics entries were finite. Full trace: `experiments/phase3d_spdi_alpha_dynamics.json`.

## Resource usage

- Total wall time: 3220.7 seconds (53.7 minutes), from training start through epoch 3 validation.
- Driver-reported peak GPU memory: 21867 MiB of 22528 MiB; training log memory peak: 17873 MiB. Reserved-memory telemetry was not recorded by the training hook.

## Inference-only GlobalShift diagnostic

| setting | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 56.39 | 40.17 | 3.50 | 41.33 | 19.24 | 29.35 | 47.68 | 27.95 |
| fixed dx=-1,dy=0 | 56.39 | 41.97 | 4.76 | 43.28 | 12.72 | 31.44 | 48.80 | 27.85 |

The fixed shift changed mAP50 by +1.80 pp and mAP75 by +1.26 pp, while tiny1 changed by -6.52 pp. This is an inference-only observation, not a trained method or causal claim.

## Decision

- **SPDI = GO** under the requested proxy rule.
- TINY benefit: YES for tiny1 in this proxy (+5.68 pp), but this is not a stable effect size from a single 3-epoch seed.
- SMALL trade-off: YES but small is only -0.19 pp relative to control.
- SPDI preserves the original P3–P7 hierarchy and reproduces a local tiny1 gain in this paired proxy; it cannot yet be claimed as a final paper result.

No further training was started. SPDI remains a proxy-validated candidate only; seed 1/2 and 12-epoch SPDI training remain unrun.
