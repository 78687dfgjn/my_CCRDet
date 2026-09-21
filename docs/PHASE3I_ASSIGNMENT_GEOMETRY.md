# Phase 3I — P2 Anchor / ATSS Assignment Geometry Audit

## Scope and guardrails

本阶段只审计 TRAIN split 的 anchor/ATSS 几何，并按预注册方案只测试 P2 base anchor side 32 px vs 16 px。没有运行 seed=1/2、12 epoch、anchor sweep、改 loss/assigner/fusion 或其他新模块。最终只运行了一个 3-epoch P2=16 proxy。

## 1. Actual anchor geometry

通过仓库实际 AnchorGenerator 生成 base anchors，而不是仅按配置算术推断：

| Level | Current P2-P6 | Candidate P2-side16 |
|---|---:|---:|
| P2 | 32 × 32 | 16 × 16 |
| P3 | 64 × 64 | 64 × 64 |
| P4 | 128 × 128 | 128 × 128 |
| P5 | 256 × 256 | 256 × 256 |
| P6 | 512 × 512 | 512 × 512 |

每个位置均为 1 个 prior。P3-P6 的 base-anchor tensor 与 grid-prior tensor 均 torch.equal=True；P2 中心坐标保持不变。详细机器可读结果见 experiments/phase3i_anchor_equivalence.json。

## 2. TRAIN-only GT geometry

训练集配对数为 4900，有效 GT 数为 54973。统计基于训练标注，经当前 pipeline 的 640×512 几何变换；没有读取 validation AP、预测或 validation 标注来设计 anchor。

| Quantity | p10 | p25 | p50 | p75 | p90 | p95 | p98 |
|---|---:|---:|---:|---:|---:|---:|---:|
| width | 6.0000 | 7.0000 | 9.0000 | 11.0000 | 14.0000 | 16.0000 | 20.0000 |
| height | 10.0000 | 12.0000 | 14.0000 | 16.0000 | 18.0000 | 20.0000 | 23.0000 |
| sqrt(area) | 8.1240 | 9.4868 | 11.2250 | 13.0384 | 15.4272 | 16.9706 | 19.4936 |
| max side | 10.0000 | 12.0000 | 14.0000 | 16.0000 | 19.0000 | 21.0000 | 24.0000 |
| min side | 6.0000 | 7.0000 | 9.0000 | 11.0000 | 13.0000 | 15.0000 | 17.0000 |
| area | 66.0000 | 90.0000 | 126.0000 | 170.0000 | 238.0000 | 288.0000 | 380.0000 |

## 3. Released evaluator size definitions

源码位置：mmdet/datasets/evaluation/coco/cocoeval.py。面积过滤条件是 area < lower or area > upper，因此上下边界均包含；相邻组在 64、144、400、1024 处存在源码层面的重叠。

| Group | Area interval |
|---|---:|
| tiny | [1, 400] |
| tiny1 | [1, 64] |
| tiny2 | [64, 144] |
| tiny3 | [144, 400] |
| small | [400, 1024] |

## 4. TRAIN assignment audit

ATSS 使用仓库实现的每层 top-k=9 候选、阈值 mean(IoU)+std(IoU)、中心内约束及最高 IoU 冲突解决；本阶段没有修改 ATSS 行为。

| Group | N | mean positives 32 | zero-positive 32 | P2 positive fraction 32 | median best IoU 32 | median threshold 32 | median best positive IoU 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny1 | 4943 | 2.2529 | 0.0577 | 1.0000 | 0.0527 | 0.0342 | 0.0508 |
| tiny2 | 30257 | 5.0882 | 0.0002 | 1.0000 | 0.1025 | 0.0664 | 0.1025 |
| tiny3 | 20526 | 7.7783 | 0.0000 | 1.0000 | 0.1826 | 0.1183 | 0.1826 |
| small | 868 | 8.9608 | 0.0000 | 1.0000 | 0.4550 | 0.2880 | 0.4550 |
| tiny | 54084 | 5.8252 | 0.0053 | 1.0000 | 0.1230 | 0.0797 | 0.1230 |

| Group | mean positives 16 | zero-positive 16 | P2 positive fraction 16 | median best IoU 16 | median threshold 16 | median best positive IoU 16 |
|---|---:|---:|---:|---:|---:|---:|
| tiny1 | 2.2535 | 0.0577 | 1.0000 | 0.2109 | 0.1146 | 0.2031 |
| tiny2 | 5.0892 | 0.0002 | 1.0000 | 0.4062 | 0.2036 | 0.4062 |
| tiny3 | 7.7275 | 0.0000 | 1.0000 | 0.6180 | 0.2843 | 0.6180 |
| small | 9.0311 | 0.0000 | 0.9830 | 0.5120 | 0.2909 | 0.5120 |
| tiny | 5.8069 | 0.0053 | 1.0000 | 0.4599 | 0.2270 | 0.4599 |

### Assignment deltas: P2=16 minus P2=32

| Group | Δ mean positives | Δ zero-positive | Δ median best IoU | Δ median threshold | Δ median best positive IoU |
|---|---:|---:|---:|---:|---:|
| tiny1 | +0.0006 | +0.0000 | +0.1582 | +0.0804 | +0.1523 |
| tiny2 | +0.0010 | +0.0000 | +0.3037 | +0.1371 | +0.3037 |
| tiny3 | -0.0508 | +0.0000 | +0.4354 | +0.1659 | +0.4354 |
| small | +0.0703 | +0.0000 | +0.0570 | +0.0030 | +0.0570 |
| tiny | -0.0182 | +0.0000 | +0.3368 | +0.1472 | +0.3368 |

关键 TRAIN-only 观察：总体 tiny 的 median best candidate IoU 从 0.1230 增至 0.4599，zero-positive rate 保持 0.0053，median best-positive IoU 同样增加 0.3368；P3-P6 先验完全不变。因此 TRAINING_GATE = PASS。该几何改善没有自动等价于验证 AP 改善。

## 5. Single pre-registered detector proxy

使用与 Phase 3B Lite-P2 完全相同的 seed=0、3 epoch、batch=8、640×512、SGD lr=0.001、semantic warm-start、数据/损失/评估协议；唯一模型差异是 P2 base anchor side 32→16。5-iteration full-resolution smoke PASS：peak allocated 18.51 GiB，peak reserved 20.06 GiB，driver-used 21621 MiB，无 NaN/Inf/OOM。

| Epoch | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 56.0100 | 33.7200 | 1.0700 | 35.0600 | 24.1500 | 25.0700 | 41.4300 | 24.3900 |
| 2 | 56.5600 | 39.8300 | 2.6500 | 41.4500 | 26.7700 | 29.7000 | 48.1300 | 22.6700 |
| 3 | 57.3000 | 41.5000 | 3.2700 | 42.8800 | 31.8400 | 30.9800 | 49.2300 | 24.9100 |

paired control 为 Phase 3B Lite-P2 P2=32 的 epoch3/best：mAP25=57.15，mAP50=41.58，mAP75=3.29，tiny=42.91，tiny1=32.22，tiny2=31.08，tiny3=49.24，small=25.40。

最佳 mAP50 为 epoch 3：mAP50=41.50。相对 paired control：

| Metric | Candidate best | Control | Δ pp |
|---|---:|---:|---:|
| mAP25 | 57.30 | 57.15 | +0.15 |
| mAP50 | 41.50 | 41.58 | -0.08 |
| mAP75 | 3.27 | 3.29 | -0.02 |
| tiny | 42.88 | 42.91 | -0.03 |
| tiny1 | 31.84 | 32.22 | -0.38 |
| tiny2 | 30.98 | 31.08 | -0.10 |
| tiny3 | 49.23 | 49.24 | -0.01 |
| small | 24.91 | 25.40 | -0.49 |

## 6. Conclusion

**TRAINING_GATE = PASS.** 预注册几何候选在 TRAIN assignment 层面满足全部四项门槛。

**ANCHOR_GEOMETRY = NO-GO（当前 proxy）。** mAP50 仅 -0.08 pp，未达到下降 NO-GO 的 -0.20 pp，但 mAP75 为 -0.02 pp 且整体基本持平，未满足正向判据；tiny 为 -0.03 pp，small 为 -0.49 pp。因此不再测试其他 anchor size，也不启动正式训练。

解释边界：P2=16 显著改善了候选 IoU 几何，但单个 3-epoch paired proxy 不支持“固定 P2 anchor 改善检测”的结论。该结果更适合作为 assignment/prior 几何已改善、但当前训练/预测链路仍未转化为验证收益的机制证据。

机器可读产物：experiments/phase3i_train_gt_geometry.json、phase3i_assignment_current32.json、phase3i_assignment_p2side16.json、phase3i_assignment_delta.json、phase3i_anchor_equivalence.json、phase3i_memory.json、phase3i_anchor16_proxy.json。
