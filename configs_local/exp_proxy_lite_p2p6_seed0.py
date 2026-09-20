_base_ = ['./exp_p2p6_seed0.py']

# Paired 3-epoch screening candidate from the semantic warm-start checkpoint.
data = dict(samples_per_gpu=8, workers_per_gpu=8)
optimizer = dict(lr=0.001)
runner = dict(max_epochs=3)
evaluation = dict(interval=1, metric='bbox')
seed = 0
deterministic = True
work_dir = '/root/CCRDet/work_dirs/phase3b_proxy_lite_p2p6_seed0'
load_from = '/hy-tmp/CCRDet_assets/phase3a_semantic_warmstart_seed0.pth'
