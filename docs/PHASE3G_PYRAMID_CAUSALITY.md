# Phase 3G 2x2 Pyramid Causality Ablation

One-seed, 3-epoch warm-start proxy comparison; descriptive only.

## Gates

- P3-P6 semantic warm-start: PASS; P2 absent and P7 discarded.
- P2-P7 semantic warm-start: PASS; P2 fresh and P7 retained.
- Full-resolution smoke: P3-P6 5 iterations PASS; P2-P7 20 iterations PASS; AMP off.
- Both new proxies completed 3 epochs with epoch-wise validation.

## Best mAP50 cells

| Topology | P2 | P7 | best epoch | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P3-P7 | 0 | 1 | 3 | 0.5635 | 0.4001 | 0.0346 | 0.4130 | 0.1356 | 0.2911 | 0.4760 | 0.2814 |
| P3-P6 | 0 | 0 | 3 | 0.5659 | 0.4031 | 0.0353 | 0.4156 | 0.1369 | 0.2923 | 0.4802 | 0.2731 |
| P2-P7 | 1 | 1 | 3 | 0.5744 | 0.4126 | 0.0311 | 0.4281 | 0.3234 | 0.3293 | 0.4941 | 0.2311 |
| P2-P6 | 1 | 0 | 3 | 0.5715 | 0.4158 | 0.0329 | 0.4291 | 0.3222 | 0.3108 | 0.4924 | 0.2540 |

## Epoch 3 cells

| Topology | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3-P7 | 0.5635 | 0.4001 | 0.0346 | 0.4130 | 0.1356 | 0.2911 | 0.4760 | 0.2814 |
| P3-P6 | 0.5659 | 0.4031 | 0.0353 | 0.4156 | 0.1369 | 0.2923 | 0.4802 | 0.2731 |
| P2-P7 | 0.5744 | 0.4126 | 0.0311 | 0.4281 | 0.3234 | 0.3293 | 0.4941 | 0.2311 |
| P2-P6 | 0.5715 | 0.4158 | 0.0329 | 0.4291 | 0.3222 | 0.3108 | 0.4924 | 0.2540 |

## Factor contrasts (percentage points)

| Contrast | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P2 effect with P7 | +1.09 | +1.25 | -0.35 | +1.51 | +18.78 | +3.82 | +1.81 | -5.03 |
| P2 effect without P7 | +0.56 | +1.27 | -0.24 | +1.35 | +18.53 | +1.85 | +1.22 | -1.91 |
| P7 effect without P2 | -0.24 | -0.30 | -0.07 | -0.26 | -0.13 | -0.12 | -0.42 | +0.83 |
| P7 effect with P2 | +0.29 | -0.32 | -0.18 | -0.10 | +0.12 | +1.85 | +0.17 | -2.29 |
| 2x2 interaction | +0.53 | -0.02 | -0.11 | +0.16 | +0.25 | +1.97 | +0.59 | -3.12 |

## Interpretation

- P2 mAP50 effect is positive with P7 (+1.25 pp) and without P7 (+1.27 pp); P2 mechanism is SUPPORTED in this proxy.
- P7 removal alone is slightly negative for mAP50 in both P2 conditions; P7 deletion alone does not explain the Lite-P2 gain.
- tiny1 improves strongly with P2 (+18.78 pp with P7; +18.53 pp without P7), while small changes -5.03 pp and -1.91 pp.
- mAP75 changes -0.35 pp with P7 and -0.24 pp without P7; high-IoU localization benefit is not established.
- BEST_PROXY_TOPOLOGY is P2-P6 by best mAP50; this remains descriptive, not a formal 12-epoch result.

## Phase 3F status

CGMA_ALIGNMENT_CLAIM = NOT ESTABLISHED. The train-only ZNCC winner (-2,+2) was at the search boundary; frozen calibration degraded strongly; and learned P2 residual alignment had no positive diagnostic signal. The 44.40 observation is retraining under train-derived fixed translation plus residual branch, not proof of geometric correction.

## Resource summary

| Smoke | status | iterations | peak allocated (GiB) | peak reserved (GiB) | driver used (MiB) |
|---|---|---:|---:|---:|---:|
| P3-P6 | PASS | 5 | 16.89 | 18.39 | 20311 |
| P2-P7 | PASS | 20 | 18.52 | 20.07 | 21635 |

## Limitations

The two new cells are one-seed, 3-epoch warm-start screening experiments. No multi-seed statistical or formal 12-epoch claim is made.
