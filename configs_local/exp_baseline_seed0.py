_base_ = ['./gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py']

model = dict(
    global_shift=dict(mode='off'),
    fusion_types=['fusion', 'fusion', 'fusion', 'fusion_cat', 'fusion_cat'])

seed = 0
deterministic = True
work_dir = '/root/CCRDet/work_dirs/phase3a_baseline_seed0'
