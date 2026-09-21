_base_ = ['./exp_proxy_lite_p2p6_seed0.py']

# Phase 3H: only P2 Fusion_CAT is replaced by PGCF.
model = dict(
    fusion_types=[
        'p2_pgcf', 'fusion', 'fusion', 'fusion', 'fusion_cat'],
    pgcf_analysis=True,
    global_shift=dict(mode='off'),
    p2_detail=dict(enabled=False),
    p2_alignment=dict(enabled=False))

work_dir = '/hy-tmp/CCRDet_assets/work_dirs/phase3h_proxy_pgcf_p2p6_seed0'
load_from = '/hy-tmp/CCRDet_assets/phase3a_semantic_warmstart_seed0.pth'
