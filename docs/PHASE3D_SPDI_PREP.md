# Phase 3D-PREP: Selective P2 Detail Injection (SPDI)

SPDI is an engineering prototype prepared for later training. This phase did not
run a formal training experiment, a proxy experiment, seed 1/2, GlobalShift, or
any new detection head/loss/assignment/augmentation.

## 1. Files changed

- `mmdet/models/utils/p2_detail_injection.py`
- `mmdet/models/detectors/afdet.py`
- `mmdet/models/utils/__init__.py`
- `configs_local/exp_spdi_seed0.py`
- `tests/test_phase3d_spdi.py`
- `experiments/phase3d_spdi_prep.json`

The existing CCRDet FPN remains `start_level=1, num_outs=5`; the detector still
returns exactly five levels with strides `[8, 16, 32, 64, 128]`.

## 2. Module structure

```text
backbone C2 RGB       ─┐
                       ├─ concat + 1x1 Fusion_CAT ─ depthwise 3x3/stride2
backbone C2 thermal   ─┘                                  └─ pointwise 1x1
                                                               = P2_detail

normal CCRDet P3 = Fusion(P3_rgb, P3_thermal) = P3_base

gate = sigmoid(Conv1x1(concat(P3_base, P2_detail)))
P3_out = P3_base + alpha * gate * P2_detail
```

SPDI reads `x[0], y[0]` directly from the backbone and does not change the
released FPN or detection-level topology. It never applies quadratic attention
to P2. P4, P5, P6, and P7 are untouched.

When `p2_detail.enabled=False` or the field is absent, the module is not created;
the released forward path and state-dict key set are preserved.

## 3. Zero-init equivalence

Using `weights/weight_rgbt.pth` and identical fixed CUDA inputs at 512x640:

- detector output levels compared: 5;
- `max_abs_diff`: `0.0`;
- `torch.equal` for every P3-P7 output: `True`;
- result: `PASS`.

`alpha` is initialized to exactly zero. The gradient test below temporarily sets
alpha to `0.1` after this identity test; otherwise the mathematically expected
zero residual scale would also zero the first-order gradients of the branch
parameters.

## 4. Gradient test

With the temporary non-zero test scale, all checks were finite and non-zero:

| tensor | result |
|---|---|
| alpha | PASS |
| depthwise convolution | PASS |
| pointwise convolution | PASS |
| gate convolution | PASS |
| original CCRDet P3 Fusion | PASS |
| loss | finite |

## 5. Tensor shapes

For input `[1,3,512,640]` per modality:

| tensor | shape |
|---|---|
| RGB C2/P2 | `[1,256,128,160]` |
| thermal C2/P2 | `[1,256,128,160]` |
| P2 detail after stride-2 branch | `[1,256,64,80]` |
| detector P3 | `[1,256,64,80]` |
| detector P4 | `[1,256,32,40]` |
| detector P5 | `[1,256,16,20]` |
| detector P6 | `[1,256,8,10]` |
| detector P7 | `[1,256,4,5]` |

The final detector output has five levels, not six.

## 6. Parameter and approximate FLOP accounting

| component | parameters |
|---|---:|
| P2 Fusion_CAT 1x1 | 131,328 |
| depthwise 3x3 | 2,560 |
| pointwise 1x1 | 65,792 |
| 1-channel gate | 513 |
| alpha | 1 |
| total | 200,194 |

At 512x640, the approximate added computation is `6,068,633,600 FLOPs`,
counting one multiply-add as two FLOPs. The estimate includes the P2 fusion,
depthwise/pointwise downsample, and spatial gate; it excludes the unchanged
backbone, FPN, and CCRDet Fusion operations.

## 7. Memory smoke

The real training data path was used for exactly five batch-8 train steps at
512x640, with AMP and gradient accumulation disabled. This is a resource smoke
test, not a training run.

| item | result |
|---|---:|
| status | PASS |
| iterations | 5 / 5 |
| peak allocated | 19,099,830,784 bytes (17.79 GiB) |
| peak reserved | 21,086,863,360 bytes (19.64 GiB) |
| peak driver-used | 21,147 MiB |
| NaN/Inf/OOM | none |

## 8. State-dict compatibility

The disabled configuration loaded the official checkpoint strictly with:

- missing keys: 0;
- unexpected keys: 0;
- checkpoint keys: 830;
- model keys: 830.

The enabled configuration had exactly nine expected missing SPDI keys and no
unexpected keys. These are newly initialized branch parameters; all original
CCRDet parameters load normally.

The complete machine-readable result is
`experiments/phase3d_spdi_prep.json`.

## 9. Readiness decision

`READY_FOR_TRAINING = YES` as an implementation gate: identity, gradients, shapes,
state loading, and the five-step resource smoke all passed. No formal training
was started in this phase. The `YES` flag does not constitute an accuracy claim
or approval to change the experimental protocol.
