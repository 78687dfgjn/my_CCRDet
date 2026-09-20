# Tiny1 failure re-analysis

This is an analysis-only re-run using the released `weight_rgbt.pth`, the
released `tools/test.py` result format, and the official tiny evaluator
settings documented in `docs/TINY_EVAL_PROTOCOL.md`.

The table is built from all category-aware tiny1 GT instances in the official
class set (`person`, `rider`, `crowd`). It uses IoU=0.50 for the formal match,
while retaining the best same-category post-NMS candidate and flags for
IoU>=0.25 and IoU>=0.50. `C` means an IoU>=0.50 candidate existed but the
official evaluator did not mark that GT as the formal TP. FPN origin is
`UNKNOWN_POST_NMS` because the returned detections no longer retain level
provenance.

## Recomputed counts

| type | definition | count |
|---|---|---:|
| A | no candidate with IoU>=0.25 | 161 |
| B | best candidate IoU in [0.25, 0.50) | 256 |
| C | candidate IoU>=0.50 but not formal TP | 0 |
| D | official IoU=0.50 TP | 112 |

Total evaluable tiny1 GT: **529**. The previous historical A/B/C numbers were
not reused; the result above was recomputed from the released checkpoint and
the official matching path.

## Interpretation boundary

The result shows that the dominant tiny1 miss group is B (near candidates that
do not reach IoU 0.50), followed by A (no candidate reaching IoU 0.25). It
does not show a ranking-only C group under this protocol. This is evidence
for a localization/overlap problem in many misses, but it is not by itself
evidence for a systematic box-size bias or for a specific FPN level.

Full per-instance records are saved under
`experiments/tiny1_reanalysis/` on the server, including image/annotation IDs,
GT box/area, formal match, highest score, best IoU, candidate box, size
ratios, and the explicit unknown FPN-level field.

The failure-type proportions among the 417 FNs are B=61.4% and A=38.6%.
This is a localization-oriented observation, not a causal attribution to a
particular neck level.
