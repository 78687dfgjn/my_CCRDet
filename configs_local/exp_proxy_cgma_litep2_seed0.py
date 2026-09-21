# Phase 3F candidate pipeline. Calibration is before RandomFlip.
_train_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
    dict(type='PairThermalCalibration', dx=-2, dy=2, fill_value=93.1),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='MultiNormalize',
         mean_list=([115.37, 121.82, 122.63], [93.10, 93.10, 93.10]),
         std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]),
         to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
_test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='PairThermalCalibration', dx=-2, dy=2,
                 fill_value=93.1),
            dict(type='RandomFlip'),
            dict(type='MultiNormalize',
                 mean_list=([115.37, 121.82, 122.63],
                            [93.10, 93.10, 93.10]),
                 std_list=([85.13, 89.01, 88.27],
                            [50.24, 50.24, 50.24]),
                 to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

_base_ = ['./exp_p2p6_seed0.py']

model = dict(
    global_shift=dict(mode='off'),
    p2_detail=dict(enabled=False),
    p2_alignment=dict(
        enabled=True,
        mode='residual',
        feature_stride=4,
        hidden_channels=64,
        max_residual_px=1.0,
        analysis_enabled=True))
data = dict(
    train=dict(pipeline=_train_pipeline),
    val=dict(pipeline=_test_pipeline),
    test=dict(pipeline=_test_pipeline))
runner = dict(max_epochs=3)
evaluation = dict(interval=1, metric='bbox')
seed = 0
deterministic = True
work_dir = '/root/CCRDet/work_dirs/phase3f_proxy_cgma_litep2_seed0'
load_from = '/hy-tmp/CCRDet_assets/phase3a_semantic_warmstart_seed0.pth'
resume_from = None

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='Phase3FAlignmentHook',
         output_json='/root/CCRDet/experiments/phase3f_offset_dynamics.json',
         iteration_points=[1, 2, 5, 10, 20, 50, 100])
]
