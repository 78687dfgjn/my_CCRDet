_base_ = ['./gfl_ccrdet_fpn_1x_rgbtdroneperson_local.py']

# Three-epoch paired screening run.  This is SPDI-v2 only.
model = dict(
    attention_mode='full',
    global_shift=dict(mode='off'),
    p2_detail=dict(
        enabled=True,
        gate_channels=1,
        zero_init=True,
        init_mode='zero_projection'))

data = dict(samples_per_gpu=8, workers_per_gpu=8)
optimizer = dict(lr=0.001)
runner = dict(max_epochs=3)
evaluation = dict(interval=1, metric='bbox')
seed = 0
deterministic = True
work_dir = '/root/CCRDet/work_dirs/phase3e_proxy_spdi_v2_seed0'
load_from = '/root/CCRDet/work_dirs/baseline_seed_0/epoch_12.pth'
resume_from = None

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(
        type='SPDIV2DynamicsHook',
        output_json='/root/CCRDet/experiments/phase3e_spdi_v2_dynamics.json',
        iteration_points=[1, 2, 5, 10, 20, 50, 100]),
]
