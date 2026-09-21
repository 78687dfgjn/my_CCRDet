# Phase 3J: P2 mechanism probe. Only the C2 lateral contribution to P2 is
# excluded; the normal FPN top-down path and all other protocol fields remain
# identical to the Phase 3B Lite-P2 proxy.
_base_ = ['./exp_proxy_lite_p2p6_seed0.py']

model = dict(
    neck=dict(p2_lateral_mode='topdown_only'),
    p2_detail=dict(enabled=False),
    p2_alignment=dict(enabled=False),
    global_shift=dict(mode='off'))

work_dir = '/hy-tmp/CCRDet_assets/work_dirs/phase3j_proxy_pseudo_p2_topdown_seed0'
load_from = '/hy-tmp/CCRDet_assets/phase3a_semantic_warmstart_seed0.pth'
