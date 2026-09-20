_base_ = ['./gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py']

model = dict(
    global_shift=dict(mode='fixed', dx=-1.0, dy=0.0),
    fusion_types=['fusion', 'fusion', 'fusion', 'fusion_cat', 'fusion_cat'])

seed = 0
deterministic = True
work_dir = '/root/CCRDet/work_dirs/phase3a_global_fixed_seed0'
