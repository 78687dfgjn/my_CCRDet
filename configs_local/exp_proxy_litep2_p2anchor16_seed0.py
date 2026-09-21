_base_ = ['./exp_proxy_lite_p2p6_seed0.py']

# Phase 3I: one pre-registered P2 anchor geometry probe.
# Only the P2 base size changes: 32 px -> 16 px.
model = dict(
    bbox_head=dict(
        anchor_generator=dict(
            base_sizes=[2, 8, 16, 32, 64])))

work_dir = '/hy-tmp/CCRDet_assets/work_dirs/phase3i_proxy_litep2_p2anchor16_seed0'
