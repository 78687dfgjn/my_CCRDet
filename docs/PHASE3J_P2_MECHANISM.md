# Phase 3J — P2 Mechanism Decomposition

## Scope

This phase used one new 3-epoch seed-0 proxy. The topdown-only P2 is a causal mechanism probe, not a proposed method. No 12-epoch or additional-seed training was run.

## Path and resource checks

- Path tests: **PASS**. P3–P6 are bitwise equal between normal and topdown-only FPN; only P2 changes.
- C2 exclusion: **PASS**. Replacing C2 leaves topdown-only P2 unchanged and changes normal P2.
- Gradient-path test: **PASS**. `neck.lateral_convs.0` has no gradient in the topdown-only P2 composition; P3–P6 gradients remain finite.
- Feature output: five levels, P2 shape `[B,256,128,160]` at 512×640 input.
- Five-iteration batch-8 smoke: **PASS**, AMP off, no OOM/NaN/Inf. Peak allocated 19867397120 bytes (None GiB), reserved 21533556736 bytes (None GiB), driver-used None MiB.

See `experiments/phase3j_path_tests.json`, `experiments/phase3j_memory.json`, and `experiments/phase3j_protocol_audit.json`.

## Candidate metrics

All AP values are percentage points.

| epoch | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 55.83 | 34.08 | 0.94 | 35.31 | 22.36 | 25.65 | 41.58 | 24.72 |
| 2.00 | 56.75 | 39.75 | 2.63 | 41.52 | 23.80 | 29.83 | 48.40 | 21.95 |
| 3.00 | 57.22 | 41.58 | 3.26 | 43.04 | 31.35 | 30.59 | 49.31 | 24.40 |

Best mAP50 epoch: **3** (41.58). Epoch-3 is also the best epoch. Training wall time was approximately **80.0 minutes**.

## Verified references

| system | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NO_P2: P3–P6 | 56.59 | 40.31 | 3.53 | 41.56 | 13.69 | 29.23 | 48.02 | 27.31 |
| PSEUDO_P2: topdown-only P2–P6 | 57.22 | 41.58 | 3.26 | 43.04 | 31.35 | 30.59 | 49.31 | 24.40 |
| TRUE_P2: normal P2–P6 | 57.15 | 41.58 | 3.29 | 42.91 | 32.22 | 31.08 | 49.24 | 25.40 |

## Decomposition

| metric | dense-grid: pseudo − no P2 | C2 detail: true − pseudo | total P2: true − no P2 | recovery fraction |
|---|---:|---:|---:|---:|
| mAP25 | +0.63 | -0.07 | +0.56 | 1.125 |
| mAP50 | +1.27 | +0.00 | +1.27 | 1.000 |
| mAP75 | -0.27 | +0.03 | -0.24 | 1.125 |
| tiny | +1.48 | -0.13 | +1.35 | 1.096 |
| tiny1 | +17.66 | +0.87 | +18.53 | 0.953 |
| tiny2 | +1.36 | +0.49 | +1.85 | 0.735 |
| tiny3 | +1.29 | -0.07 | +1.22 | 1.057 |
| small | -2.91 | +1.00 | -1.91 | 1.524 |

For mAP50, the dense-grid contribution is **+1.27 pp**, the C2-detail contribution is **+0.00 pp**, and the recovery fraction is **1.000**. For tiny1, the dense-grid contribution is **+17.66 pp**, the C2-detail contribution is **+0.87 pp**, and the recovery fraction is **0.953**.

## Interpretation

Under the pre-registered 75%/25% rule, the current one-seed proxy labels the primary mechanism **`P2_PRIMARY_MECHANISM = DENSE_PREDICTION_GRID`**. This is descriptive evidence from a single 3-epoch proxy, not a causal claim about all training regimes.

The mAP75 pattern does not show a high-IoU benefit: the dense-grid probe is already -0.27 pp relative to no P2, while C2 detail adds only +0.03 pp. The small-object trade-off is also already present in the dense-grid probe (-2.91 pp), although the normal C2 path recovers +1.00 pp and leaves a -1.91 pp net difference.

No additional training was started. The server remains running.
