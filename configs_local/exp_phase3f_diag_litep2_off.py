_base_ = ['./exp_p2p6_seed0.py']
model = dict(global_shift=dict(mode='off'),
             p2_detail=dict(enabled=False),
             p2_alignment=dict(enabled=False))
work_dir = '/root/CCRDet/work_dirs/phase3f_diag_litep2_off'
load_from = '/hy-tmp/CCRDet_assets/work_dirs/phase3b_proxy_lite_p2p6_seed0/epoch_3.pth'
resume_from = None
