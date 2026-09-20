dataset_type = 'DronePerson'
data_root = '/home/gpz/data/RGBTDronePerson/'
img_norm_cfg = dict(
    mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
    std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]),
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='MultiNormalize',
        mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
        std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]),
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='MultiNormalize',
                mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
                std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]),
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='DronePerson',
        ann_file='/root/CCRDet/train_thermal.json',
        img_prefix='/root/CCRDet/train/',
        pipeline=[
            dict(
                type='LoadImagePairFromFile',
                spectrals=('visible', 'thermal')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='MultiNormalize',
                mean_list=([115.37, 121.82, 122.63], [93.1, 93.1, 93.1]),
                std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24]),
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='DronePerson',
        ann_file='/root/CCRDet/val_thermal.json',
        img_prefix='/root/CCRDet/val/',
        pipeline=[
            dict(
                type='LoadImagePairFromFile',
                spectrals=('visible', 'thermal')),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='MultiNormalize',
                        mean_list=([115.37, 121.82,
                                    122.63], [93.1, 93.1, 93.1]),
                        std_list=([85.13, 89.01, 88.27], [50.24, 50.24,
                                                          50.24]),
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='DronePerson',
        ann_file='/root/CCRDet/val_thermal.json',
        img_prefix='/root/CCRDet/val/',
        pipeline=[
            dict(
                type='LoadImagePairFromFile',
                spectrals=('visible', 'thermal')),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='MultiNormalize',
                        mean_list=([115.37, 121.82,
                                    122.63], [93.1, 93.1, 93.1]),
                        std_list=([85.13, 89.01, 88.27], [50.24, 50.24,
                                                          50.24]),
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='GFLAF',
    backbone=dict(
        type='AMFusionResNet',
        dims_in=[256, 512, 1024, 2048],
        dims_lsk=[32, 32, 64, 64],
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        pretrained='pretrain_weights/resnet50-2stream.pth'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.3),
        max_per_img=100),
    tanh=False,
    fusion_types=['fusion_cat', 'fusion', 'fusion', 'fusion', 'fusion_cat'],
    global_shift=dict(mode='off'),
    p2_detail=dict(enabled=False))
work_dir = '/root/CCRDet/work_dirs/phase3c_full_lite_p2p6_seed0'
seed = 0
deterministic = True
