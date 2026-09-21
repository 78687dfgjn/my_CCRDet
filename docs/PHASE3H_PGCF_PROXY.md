# Phase 3H PGCF Paired Proxy

## Status

PGCF = RESOURCE BLOCKED.

The required batch-8, 512x640, AMP-off, 20-iteration memory smoke test reached an OOM on iteration 5. nvidia-smi was checked after the failure: the RTX 2080 Ti had no unrelated GPU process. Per the pre-registered rule, batch size, resolution, attention behavior, and precision were not changed, and the 3-epoch proxy was not started.

## Identity and warm-start gates

- Identity test: PASS.
- All P2-P6 features were torch.equal with max absolute difference 0.0.
- Gate-logit initial weight and bias norms: 0.0, 0.0.
- Warm-start compatibility: True.
- Gradient test: PASS. Iteration 1 gate-logit gradient was nonzero; reduce/depthwise gradients were zero as allowed. Both became nonzero at iteration 2.

## PGCF accounting

| Component | Parameters |
|---|---:|
| Existing P2 Fusion_CAT | 131328 |
| PGCF total | 157312 |
| Additional | 25984 |
| reduce | 24608 |
| depthwise | 320 |
| gate_logits | 1056 |

At P2=128x160, the approximate additional conv-only cost is 1060372480 FLOPs; abs, ReLU, softmax and reshaping are excluded.

## Paired control

| Metric | Lite-P2 Fusion_CAT control | PGCF | Delta |
|---|---:|---:|---:|
| mAP25 | 0.5715 | N/A | N/A |
| mAP50 | 0.4158 | N/A | N/A |
| mAP75 | 0.0329 | N/A | N/A |
| tiny | 0.4291 | N/A | N/A |
| tiny1 | 0.3222 | N/A | N/A |
| tiny2 | 0.3108 | N/A | N/A |
| tiny3 | 0.4924 | N/A | N/A |
| small | 0.2540 | N/A | N/A |

The control reference is the existing Phase 3B Lite-P2-P6 epoch-3/best proxy. No candidate AP, gate trajectory by epoch, or PGCF decision based on accuracy can be reported because the resource gate failed first.

## Smoke resources

| Item | Value |
|---|---:|
| Completed iterations | 4 |
| Attempted blocking iteration | 5 |
| Peak allocated | 19.54 GiB |
| Peak reserved | 20.22 GiB |
| Peak driver used | 22143 MiB |
| Status | RESOURCE_BLOCKED |

Phase 3H stops here. No 12-epoch run, second seed, alternate PGCF, alignment, SPDI, GlobalShift, loss or assigner changes were started. The server remains running.
