_base_ = ['./gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py']

model = dict(
    neck=dict(
        start_level=0,
        num_outs=5),
    bbox_head=dict(
        anchor_generator=dict(strides=[4, 8, 16, 32, 64])),
    global_shift=dict(mode='off'),
    fusion_types=['fusion', 'fusion', 'fusion', 'fusion_cat', 'fusion_cat'])

optimizer = dict(lr=0.001)
runner = dict(max_epochs=3)
seed = 0
deterministic = True
work_dir = '/root/CCRDet/work_dirs/phase3a_p2p6_seed0'
load_from = '/hy-tmp/CCRDet_assets/phase3a_p2p6_warmstart_seed0.pth'
