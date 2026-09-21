# Phase 3G D: P2-P7 proxy (P2 present, P7 present).
_base_ = ['./gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py']

model = dict(
    neck=dict(start_level=0, num_outs=6),
    bbox_head=dict(
        anchor_generator=dict(strides=[4, 8, 16, 32, 64, 128])),
    fusion_types=[
        'fusion_cat', 'fusion', 'fusion', 'fusion',
        'fusion_cat', 'fusion_cat'],
    attention_mode='full',
    global_shift=dict(mode='off'),
    p2_detail=dict(enabled=False),
    p2_alignment=dict(enabled=False))

data = dict(samples_per_gpu=8, workers_per_gpu=8)
optimizer = dict(lr=0.001)
runner = dict(max_epochs=3)
evaluation = dict(interval=1, metric='bbox')
seed = 0
deterministic = True
work_dir = '/hy-tmp/CCRDet_assets/work_dirs/phase3g_proxy_p2p7_seed0'
load_from = '/hy-tmp/CCRDet_assets/phase3g_warmstart_p2p7_seed0.pth'
