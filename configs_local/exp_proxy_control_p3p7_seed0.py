_base_ = ['./gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py']

# Paired 3-epoch screening control from the same epoch-12 baseline checkpoint.
data = dict(samples_per_gpu=8, workers_per_gpu=8)
optimizer = dict(lr=0.001)
runner = dict(max_epochs=3)
evaluation = dict(interval=1, metric='bbox')
seed = 0
deterministic = True
work_dir = '/root/CCRDet/work_dirs/phase3b_proxy_control_p3p7_seed0'
load_from = '/root/CCRDet/work_dirs/baseline_seed_0/epoch_12.pth'
