_base_ = [
    './gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py'
]

# Configuration-only candidate.  No detector/backbone/fusion implementation
# is changed here.  The current GFLAF detector still hard-codes five fusion
# modules; see docs/TRAINING_PLAN_P2.md before attempting to train this file.
model = dict(
    neck=dict(
        start_level=0,
        num_outs=6),
    bbox_head=dict(
        anchor_generator=dict(
            strides=[4, 8, 16, 32, 64, 128])))

seed = 0
deterministic = True
work_dir = '/root/CCRDet/work_dirs/ccrdet_p2_p7_seed0'
