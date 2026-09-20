# Phase 2C hypothesis falsification report

No training was started. Seed 1/2 remain paused, and no P2 refinement,
learnable alignment, local alignment, TDS, or JDFF module was implemented.

## 1. Official evaluation protocol

The released call stack uses model-side score threshold 0.05, NMS IoU 0.30,
`max_per_img=100`, followed by the released tiny evaluator with
`maxDets=[100,300,1000]` and final evaluator maxDets=1000. The source-level
`Params` default of 200 is overwritten by `DronePerson.evaluate`; it is not the
official `tools/test.py` protocol. See `docs/TINY_EVAL_PROTOCOL.md`.

## 2. Tiny1 failure result

The official checkpoint has 529 evaluable tiny1 GT instances:

| type | count | interpretation |
|---|---:|---|
| A | 161 | no same-category candidate reaches IoU 0.25 |
| B | 256 | best candidate reaches IoU 0.25 but not 0.50 |
| C | 0 | IoU>=0.50 candidate exists but is not the formal TP |
| D | 112 | formal IoU=0.50 TP |

Thus B is the largest failure type (61.4% of FNs), followed by A (38.6%).
There is no C group under the official result and matching path. FPN origin
cannot be recovered from post-NMS detections, so no level-specific claim is
made.

## 3. Cross-checkpoint global shift

Thermal-only integer shifts were evaluated with the same released protocol;
`dx=-1` means the thermal content is shifted one network-input pixel left.

| checkpoint | dx=-2 | dx=-1 | dx=0 | dx=+1 | dx=+2 | best dx |
|---|---:|---:|---:|---:|---:|---:|
| official | 0.3783 | 0.4406 | 0.4361 | 0.3638 | 0.2326 | -1 |
| seed=1001937037 | 0.3897 | 0.4443 | 0.4315 | 0.3605 | 0.2241 | -1 |
| seed=0 | 0.3747 | 0.4214 | 0.4032 | 0.3148 | 0.1904 | -1 |
| historical epoch10 | 0.3824 | 0.4332 | 0.4241 | 0.3520 | 0.2197 | -1 |

For mAP50, the `dx=-1` improvement over `dx=0` is positive for all four
checkpoints: mean **+0.01115**, population standard deviation **0.00502**.
The per-checkpoint gains are +0.0045, +0.0128, +0.0182, and +0.0091.
The mean mAP25 change is -0.00183, while the mean mAP75 change is +0.01123;
the observed effect is therefore more visible at mAP75 than at mAP25.

This is strong cross-checkpoint evidence on the same released validation set,
but not yet a sequence-held-out generalization result.

## 4. Sequence-held-out validation

The released validation metadata contains only numeric image filenames/IDs
(`04900.jpg` to `06124.jpg`) and image dimensions. No sequence/video/frame
grouping field is available. A non-leaking DEV/HOLDOUT split could not be
recovered, and no random split was fabricated. Held-out generalization is
therefore unresolved.

## 5. P2 evidence and training plan

Historical P2 numbers remain provisional: the archived comparisons changed
more than one topology/protocol detail in some cases, and do not form a clean
paired seed-0 P3-P7 vs P2-P7 comparison. The requested configuration-only
candidate is recorded in `docs/TRAINING_PLAN_P2.md`, but the current detector
hard-codes five fusion modules and cannot safely consume six FPN outputs
without a separately reviewed compatibility change. No P2 training was run.

## 6. Decision

| hypothesis | decision | reason |
|---|---|---|
| GLOBAL | NEED MORE EVIDENCE | dx=-1 is consistent across four checkpoints, but no sequence-held-out split is available |
| P2 | NO-GO | historical evidence is not cleanly causal and the candidate is not currently runnable by config alone |
| TINY REFINEMENT | NEED MORE EVIDENCE | B-type overlap failures are common, but FPN origin and systematic size bias are unproven |

The most defensible next evidence target is sequence-aware validation or an
independent split for the fixed shift, not a new learned alignment module.
