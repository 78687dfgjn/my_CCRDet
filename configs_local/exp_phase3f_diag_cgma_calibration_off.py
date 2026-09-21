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
work_dir = '/root/CCRDet/work_dirs/phase3f_diag_cgma_calibration_off'
load_from = None
resume_from = None
