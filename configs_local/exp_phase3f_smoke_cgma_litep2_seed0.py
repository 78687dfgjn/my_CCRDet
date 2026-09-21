_base_ = ['./exp_proxy_cgma_litep2_seed0.py']

runner = dict(type='IterBasedRunner', max_epochs=None, max_iters=20)
workflow = [('train', 1)]
evaluation = None
checkpoint_config = None
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='Phase3FAlignmentHook',
         output_json='/root/CCRDet/experiments/phase3f_offset_smoke.json',
         iteration_points=[1, 2, 5, 10, 20])
]
work_dir = '/root/CCRDet/work_dirs/phase3f_smoke_cgma_litep2_seed0'
