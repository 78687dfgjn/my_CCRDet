# Phase 3A semantic warm-start correction

This change is code-only. No 3-epoch proxy training was started.

## Semantic mapping

| baseline semantic level | old index | candidate semantic level | new index | action |
|---|---:|---|---:|---|
| P3 | 0 | P3 | 1 | load/remap |
| P4 | 1 | P4 | 2 | load/remap |
| P5 | 2 | P5 | 3 | load/remap |
| P6 | 3 | P6 | 4 | load/remap |
| P7 | 4 | — | — | discard |
| — | — | P2 | 0 | fresh initialize |

The mapping is applied explicitly to RGB/T FPN lateral and output convolutions
and to fusion modules. Fusion uses the same old-index to new-index mapping;
new `fuse.0` is fresh and old `fuse.4` is discarded.

Candidate fusion types are:

```text
[Fusion_CAT, Fusion, Fusion, Fusion, Fusion_CAT]
```

## Generated checkpoint audit

Source: `work_dirs/baseline_seed_0/epoch_12.pth`

- semantically loaded keys: 824
- newly initialized keys: 10
- P2 newly initialized keys: 10
- P2 newly initialized scalar parameters: 1,443,072
- discarded source keys: 6
- semantic state-dict sanity: **PASS**

The machine-readable records are:

- `experiments/phase3a_semantic_warmstart_report.json`
- `experiments/phase3a_semantic_warmstart_sanity.json`

## Validation

- Python compilation: PASS.
- Official config vs local baseline config: strict state-dict load with no
  missing/unexpected keys; feature max absolute difference `0.0`.
- Lite-P2 input `[1,3,128,160]`: outputs
  `[P2 1x256x32x40, P3 1x256x16x20, P4 1x256x8x10,
  P5 1x256x4x5, P6 1x256x2x3]`; PASS.
- Attention guard at `[1,256,128,160]` with threshold 4096: exact `rgb + thermal`
  fallback; peak allocated 120,072,192 bytes and peak reserved 127,926,272 bytes
  in the isolated guard test. A candidate feature-only forward at 512x640
  completed with peak allocated 450,907,648 bytes and peak reserved
  528,482,304 bytes (batch 1, inference-only).

The guard is opt-in for candidate configs (`attention_mode='skip'`). The default
when `attention_mode` is omitted remains `full`, so the official baseline path is
numerically unchanged.
