"""Phase 3D-PREP: SPDI prototype, training intentionally not started."""

_base_ = ['./gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py']

model = dict(
    # Keep the released P3-P7 detection topology and strides unchanged.
    neck=dict(start_level=1, num_outs=5),
    bbox_head=dict(anchor_generator=dict(strides=[8, 16, 32, 64, 128])),
    p2_detail=dict(enabled=True, gate_channels=1, zero_init=True))

seed = 0
deterministic = True
work_dir = '/root/CCRDet/work_dirs/phase3d_spdi_seed0'
