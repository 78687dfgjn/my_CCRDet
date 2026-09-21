# Phase 3H-R PGCF Memory-Only Repair

## Final status

**PGCF resource gate: PASS. PGCF accuracy status: NO-GO.**

Activation checkpointing made the exact PGCF computation fit the required batch-8, 512x640, AMP-off 20-iteration smoke. The single permitted 3-epoch proxy then completed without OOM. The accuracy result was essentially flat against the paired Lite-P2 Fusion_CAT control (+0.03 percentage points mAP50), so no accuracy claim is supported.

## Scope and invariants

- Only autograd activation retention in PGCF was changed; PGCF mathematics, groups=16, hidden=32, gate, softmax, P2-P6 topology, optimizer, loss, input size, batch size, and AMP setting were unchanged.
- Checkpointing is enabled only in training and is inactive in evaluation.
- No 12-epoch run, seed 1/2, alternate PGCF, GlobalShift, CGMA, SPDI, alignment, loss, or assigner experiment was started.

## Checkpoint equivalence

| Check | Result |
|---|---:|
| Forward output equality | True |
| Forward max abs diff | 0.0e+00 |
| Maximum compared gradient diff | 0.0e+00 |
| Telemetry double-counting detected | False |
| Zero-logit identity with Fusion_CAT | True |

The direct checkpoint-on/off comparison was exact for the tested output and all four compared gradients. Reduce/depthwise gradients are zero at the initial zero-logit step and become nonzero at iteration 2 in the existing gradient test, as allowed by the registered test.

## Telemetry audit

The earlier group means near 0.0625 were an analysis aggregation bug: group sums were divided by the total number of modality-weight elements instead of the per-group count. Forward mathematics was unchanged. After fixing telemetry, zero logits give overall RGB/T means of 1.0 and all 16 group means of 1.0; maximum group error is 0.

## Memory smoke

| Run | Completed | Peak allocated | Peak reserved | Peak driver used | Mean iteration |
|---|---:|---:|---:|---:|---:|
| Original PGCF, no checkpoint | 4 (OOM at 5) | 19.54 GiB | 20.22 GiB | 22143 MiB | 2.438 s |
| PGCF with checkpoint | 20 | 18.51 GiB | 20.17 GiB | 22135 MiB | 2.259 s |

Checkpointed peak allocated memory was 1.03 GiB lower; reserved memory was 0.05 GiB lower and driver peak was 8 MiB lower. Timing is not a controlled overhead comparison because the original run stopped after four iterations while the repaired run completed twenty.

## 3-epoch paired proxy

| Epoch | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 55.96 | 33.90 | 0.97 | 35.18 | 24.38 | 24.94 | 41.42 | 25.00 | 1.1372 |
| 2 | 56.50 | 39.84 | 2.70 | 41.41 | 26.11 | 29.98 | 48.00 | 23.20 | 1.1055 |
| 3 | 57.19 | 41.61 | 3.32 | 42.99 | 31.96 | 31.28 | 49.26 | 24.97 | 1.0732 |

Paired control (Phase 3B Lite-P2 Fusion_CAT, epoch 3/best): mAP25=57.15, mAP50=41.58, mAP75=3.29, tiny=42.91, tiny1=32.22, tiny2=31.08, tiny3=49.24, small=25.40.

| Best epoch | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 57.19 | 41.61 | 3.32 | 42.99 | 31.96 | 31.28 | 49.26 | 24.97 |

| Best PGCF - control | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| delta (pp) | +0.04 | +0.03 | +0.03 | +0.08 | -0.26 | +0.20 | +0.02 | -0.43 |

## Gate dynamics (evaluation-only telemetry)

Each row uses 20 validation batches, with checkpoint recomputation inactive:

| Epoch | mean RGB | mean thermal | std RGB | std thermal | mean abs(weight-1) | RGB>1.25 | thermal>1.25 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.995746 | 1.004254 | 0.004937 | 0.004937 | 0.004481 | 0.0000 | 0.0000 |
| 2 | 0.995382 | 1.004618 | 0.005740 | 0.005740 | 0.004997 | 0.0000 | 0.0000 |
| 3 | 0.995587 | 1.004413 | 0.006065 | 0.006065 | 0.004951 | 0.0000 | 0.0000 |

The learned gate stayed very close to identity in this short proxy: no sampled weight exceeded 1.25, while the modality means remained near one. These are descriptive statistics, not evidence that the gate is useless beyond this run.

## Accounting and reproducibility

- PGCF total parameters: 157,312; existing P2 Fusion_CAT: 131,328; additional: 25,984.
- Approximate additional P2 convolution-only cost: 1,060,372,480 FLOPs at 128x160; abs, ReLU, softmax and reshaping excluded.
- Training wall time from the run log: 1:20:32.
- Artifacts: `phase3h_checkpoint_equivalence.json`, `phase3h_memory_checkpointed.json`, `phase3h_telemetry_audit.json`, `phase3h_gate_dynamics.json`, and `phase3h_pgcf_proxy.json`.

## Decision

**PGCF = NO-GO for further accuracy development in this phase.** The resource repair succeeded, but the paired 3-epoch accuracy delta was only +0.03 pp mAP50, with tiny +0.08 pp, mAP75 +0.03 pp, and small -0.43 pp. Per the preregistered rule, no PGCF-v2 or follow-up variant is started.
