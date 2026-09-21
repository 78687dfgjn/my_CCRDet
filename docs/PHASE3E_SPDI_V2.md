# Phase 3E — SPDI-v2 Optimization Fix + Paired Proxy

本阶段是 3-epoch paired screening，不是正式论文结果。未运行 12 epoch、seed=1/2、GlobalShift、新 loss 或新模块。

## 1. v1 optimization issue

SPDI-v1 使用 `alpha=0` 的 identity 参数化。已有 v1 trace 显示 iter1 分支梯度为 0，epoch3 alpha 约 `9.269e-05`，分支梯度约 `1e-6`。这支持“alpha 长期抑制分支优化”的待验证假设，但不单独证明因果。

## 2. v2 initialization and identity

- `init_mode=zero_projection`：alpha 初始化为 1.0；pointwise weight/bias 严格初始化为 0。
- source: `baseline_seed_0/epoch_12.pth`; no semantic warm-start and no optimizer resume.
- v2 missing keys: 9, all under `p2_detail_injection.*`; unexpected keys: 0.
- fixed official-checkpoint input: P3–P7 all `torch.equal=True`, max absolute difference 0.0.
- identity test: **PASS**.

## 3. 100-iteration optimization precheck

Precheck 在同一次训练进程内通过后继续到 3 epochs。所有记录 finite。

| point | alpha | alpha grad | pointwise weight norm | pointwise grad | depthwise grad | P2 fusion grad | gate grad | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 1.5763392e-08 | 0.016206125 | 0 | 0 | 0 | 0.905920 |
| 2 | 1 | 1.3985979e-11 | 9.1722825e-08 | 0.024978859 | 1.1871673e-11 | 1.0180148e-10 | 4.4211657e-10 | 1.010562 |
| 5 | 1 | 2.4707969e-10 | 7.9414667e-07 | 0.022685755 | 2.9805361e-10 | 2.4082771e-09 | 1.1971365e-08 | 0.872422 |
| 10 | 1 | -1.9085526e-08 | 3.8797671e-06 | 0.026829185 | 6.0115357e-09 | 2.9144029e-08 | 1.3221889e-07 | 0.989811 |
| 20 | 0.99999988 | 2.949956e-08 | 1.5514006e-05 | 0.026759949 | 1.6454981e-08 | 1.125858e-07 | 6.5449774e-07 | 0.929191 |
| 50 | 0.99999779 | 4.6167408e-08 | 6.950945e-05 | 0.02103835 | 3.9007695e-08 | 3.1338885e-07 | 2.2791742e-06 | 0.943717 |
| 100 | 0.99999034 | -4.9165942e-07 | 0.00021804136 | 0.027862919 | 2.0163752e-07 | 1.2385965e-06 | 5.4172961e-06 | 0.994444 |

结论：iter1 pointwise gradient 非零；iter2 depthwise 与 P2 fusion gradient 非零；iter100 pointwise weight norm 已为 `2.1804e-4`，因此 v2 optimization precheck **PASS**。

## 4. Proxy metrics

| epoch | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small | loss_cls | loss_bbox | loss_dfl | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 56.68 | 38.33 | 2.74 | 40.16 | 18.94 | 27.83 | 46.91 | 24.16 | 0.1714 | 0.5827 | 0.1608 | 0.9149 |
| 2 | 55.79 | 38.05 | 3.16 | 39.49 | 13.71 | 27.04 | 46.31 | 25.35 | 0.1698 | 0.5772 | 0.1604 | 0.9074 |
| 3 | 56.41 | 39.96 | 3.51 | 41.46 | 13.86 | 28.72 | 48.04 | 26.57 | 0.1651 | 0.5646 | 0.1592 | 0.8889 |

best mAP50 epoch: **3**, mAP50 **39.96**. Epoch3 mAP50 **39.96**.

## 5. CONTROL vs SPDI-v1 vs SPDI-v2

| system (best/epoch3) | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CONTROL epoch3 | 56.35 | 40.01 | 3.46 | 41.30 | 13.56 | 29.11 | 47.60 | 28.14 |
| SPDI-v1 epoch3 | 56.39 | 40.17 | 3.50 | 41.33 | 19.24 | 29.35 | 47.68 | 27.95 |
| SPDI-v2 epoch3 | 56.41 | 39.96 | 3.51 | 41.46 | 13.86 | 28.72 | 48.04 | 26.57 |

**SPDI-v2 − CONTROL (epoch3):** mAP25 +0.06, mAP50 -0.05, mAP75 +0.05, tiny +0.16, tiny1 +0.30, tiny2 -0.39, tiny3 +0.44, small -1.57.
**SPDI-v2 − SPDI-v1 (epoch3):** mAP25 +0.02, mAP50 -0.21, mAP75 +0.01, tiny +0.13, tiny1 -5.38, tiny2 -0.63, tiny3 +0.36, small -1.38.

## 6. Resource usage

- GPU: 1 × NVIDIA GeForce RTX 2080 Ti (22528 MiB).
- Wall time: approximately **3218 s (53.6 min)** from launch to epoch3 validation.
- Training log peak allocated memory: **17873 MiB**.
- An `nvidia-smi` observation before training showed 21865 MiB in use; automated driver peak and reserved-memory telemetry were not recorded, so they are not claimed as exact peaks.
- AMP and gradient accumulation: off.

## 7. Decision

按用户给定规则，SPDI-v2 相对 paired CONTROL 的 best/epoch3 mAP50 为 `39.96 - 40.01 = -0.05 pp`，不满足 +0.1 pp 门槛。

- **SPDI-V2 = NOT SUPPORTED BY CURRENT INJECTION DESIGN**（当前 3-epoch proxy）。
- TINY BENEFIT = **YES**（epoch3 tiny +0.16 pp；tiny1 +0.30 pp，但幅度很小且未跨 seed 验证）。
- SMALL TRADE-OFF = **YES**（epoch3 small -1.57 pp）。
- HIGH-IOU BENEFIT = **NO**（epoch3 mAP75 3.51 vs control 3.46（仅 +0.05 pp，不构成明确的 high-IoU gain））。

这只是当前 injection design 的低成本筛选结论；不把 subgroup 的单次 proxy 波动写成稳定效应。

## 8. Artifacts

- `experiments/phase3e_spdi_v2_proxy.json`
- `experiments/phase3e_spdi_v2_dynamics.json`
- `experiments/phase3e_spdi_v2_identity.json`
- `experiments/phase3e_protocol_audit.json`
- `configs_local/exp_proxy_spdi_v2_seed0.py`
- `tools/phase3e_spdi_v2_identity.py

