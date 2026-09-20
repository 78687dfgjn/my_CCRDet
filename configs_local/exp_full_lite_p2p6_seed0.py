"""Phase 3C formal Lite-P2-P6 training configuration.

This config starts from the released two-stream pretrained initialization. It
does not load a baseline checkpoint or a Phase 3A/3B warm-start artifact.
"""

_base_ = ['./gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py']

model = dict(
    neck=dict(start_level=0, num_outs=5),
    bbox_head=dict(anchor_generator=dict(strides=[4, 8, 16, 32, 64])),
    fusion_types=['fusion_cat', 'fusion', 'fusion', 'fusion', 'fusion_cat'],
    global_shift=dict(mode='off'),
    p2_detail=dict(enabled=False))

# Explicitly prevent accidental checkpoint continuation.
load_from = None
resume_from = None
seed = 0
deterministic = True
work_dir = '/root/CCRDet/work_dirs/phase3c_full_lite_p2p6_seed0'
