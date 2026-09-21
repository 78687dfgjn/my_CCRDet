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
model = dict(global_shift=dict(mode='off'),
             p2_detail=dict(enabled=False),
             p2_alignment=dict(enabled=False))
data = dict(
    train=dict(pipeline=_train_pipeline),
    val=dict(pipeline=_test_pipeline),
    test=dict(pipeline=_test_pipeline))
work_dir = '/root/CCRDet/work_dirs/phase3f_diag_litep2p6_calibration_on'
load_from = '/hy-tmp/CCRDet_assets/work_dirs/phase3b_proxy_lite_p2p6_seed0/epoch_3.pth'
resume_from = None
