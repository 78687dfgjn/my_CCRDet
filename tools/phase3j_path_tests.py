import hashlib, json
from pathlib import Path
import torch
from mmcv import Config
from mmdet.models import build_detector, build_neck

SOURCE = '/hy-tmp/CCRDet_assets/phase3a_semantic_warmstart_seed0.pth'
NORMAL = 'configs_local/exp_proxy_lite_p2p6_seed0.py'
TOPDOWN = 'configs_local/exp_proxy_pseudo_p2_topdown_seed0.py'
OUTPUT = 'experiments/phase3j_path_tests.json'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def make_inputs(cfg):
    chans = list(cfg.model.neck.in_channels)
    sizes = [(128,160),(64,80),(32,40),(16,20)]
    return [torch.randn(1,c,h,w) for c,(h,w) in zip(chans,sizes)]

def manual_p2(neck, inputs, zero_c2):
    lat = [conv(inputs[i + neck.start_level])
           for i,conv in enumerate(neck.lateral_convs)]
    if zero_c2:
        lat[0] = lat[0].new_zeros(lat[0].shape)
    for i in range(len(lat)-1,0,-1):
        if 'scale_factor' in neck.upsample_cfg:
            lat[i-1] = lat[i-1] + torch.nn.functional.interpolate(
                lat[i], **neck.upsample_cfg)
        else:
            lat[i-1] = lat[i-1] + torch.nn.functional.interpolate(
                lat[i], size=lat[i-1].shape[2:], **neck.upsample_cfg)
    return neck.fpn_convs[0](lat[0])

torch.manual_seed(0)
src = torch.load(SOURCE, map_location='cpu')['state_dict']
ncfg = Config.fromfile(NORMAL)
tcfg = Config.fromfile(TOPDOWN)
ncfg.model.backbone.pretrained = None
tcfg.model.backbone.pretrained = None
nm = build_detector(ncfg.model, train_cfg=ncfg.get('train_cfg'), test_cfg=ncfg.get('test_cfg'))
tm = build_detector(tcfg.model, train_cfg=tcfg.get('train_cfg'), test_cfg=tcfg.get('test_cfg'))
nm.init_weights()
tm.init_weights()
ni = nm.load_state_dict(src, strict=True)
ti = tm.load_state_dict(src, strict=True)
nnk = build_neck(ncfg.model.neck)
tnk = build_neck(tcfg.model.neck)
tnk.load_state_dict(nnk.state_dict(), strict=True)
nnk.eval()
tnk.eval()
inputs = make_inputs(ncfg)
with torch.no_grad():
    out_n = nnk(inputs)
    out_t = tnk(inputs)
    p2_n_manual = manual_p2(nnk, inputs, False)
    p2_t_manual = manual_p2(tnk, inputs, True)
p3_p6_equal = all(torch.equal(a,b) for a,b in zip(out_n[1:5], out_t[1:5]))
manual_n_equal = torch.equal(out_n[0], p2_n_manual)
manual_t_equal = torch.equal(out_t[0], p2_t_manual)
changed = [x.clone() for x in inputs]
changed[0] = torch.randn_like(changed[0])
with torch.no_grad():
    out_t_changed = tnk(changed)
    out_n_changed = nnk(changed)
td_c2_excluded = all(torch.equal(a,b) for a,b in zip(out_t,out_t_changed))
normal_c2_changes = not torch.equal(out_n[0], out_n_changed[0])
gnk = build_neck(tcfg.model.neck)
gnk.load_state_dict(nnk.state_dict(), strict=True)
gnk.train()
ginputs = [x.detach().clone().requires_grad_(True) for x in inputs]
gout = gnk(ginputs)
sum(x.sum() for x in gout).backward()
c2_grads = [None if p.grad is None else float(p.grad.abs().max())
            for p in gnk.lateral_convs[0].parameters()]
c2_zero = all(x is None or x == 0.0 for x in c2_grads)
finite = True
non_c2_max = []
for mod in list(gnk.lateral_convs)[1:] + list(gnk.fpn_convs):
    for p in mod.parameters():
        if p.grad is not None:
            finite = finite and bool(torch.isfinite(p.grad).all().item())
            non_c2_max.append(float(p.grad.abs().max()))
report = {
    'status': 'PASS' if all([
        not ni.missing_keys, not ni.unexpected_keys,
        not ti.missing_keys, not ti.unexpected_keys,
        p3_p6_equal, list(out_t[0].shape)==[1,256,128,160],
        len(out_t)==5, manual_n_equal, manual_t_equal,
        td_c2_excluded, normal_c2_changes, c2_zero, finite
    ]) else 'FAIL',
    'source_checkpoint': SOURCE,
    'source_checkpoint_sha256': sha256(SOURCE),
    'strict_state_dict': {
        'normal_missing_keys': list(ni.missing_keys),
        'normal_unexpected_keys': list(ni.unexpected_keys),
        'topdown_missing_keys': list(ti.missing_keys),
        'topdown_unexpected_keys': list(ti.unexpected_keys),
        'normal_model_state_key_count': len(nm.state_dict()),
        'topdown_model_state_key_count': len(tm.state_dict())
    },
    'feature_path': {
        'normal_mode': 'C2 lateral + top-down',
        'topdown_only_mode': 'zero C2 lateral + top-down',
        'p3_p6_torch_equal': p3_p6_equal,
        'p2_normal_shape': list(out_n[0].shape),
        'p2_topdown_shape': list(out_t[0].shape),
        'detection_level_count': len(out_t),
        'normal_manual_p2_equal': manual_n_equal,
        'topdown_manual_p2_equal': manual_t_equal,
        'p2_normal_equals_topdown': torch.equal(out_n[0],out_t[0])
    },
    'c2_exclusion': {
        'topdown_only_unchanged_after_c2_replacement': td_c2_excluded,
        'normal_p2_changes_after_c2_replacement': normal_c2_changes,
        'all_topdown_outputs_equal_after_c2_replacement': td_c2_excluded
    },
    'gradient_path': {
        'c2_lateral_parameter_max_grads': c2_grads,
        'c2_lateral_zero_or_none': c2_zero,
        'p3_p6_path_gradients_finite': finite,
        'non_c2_gradient_max_values': non_c2_max[:20]
    },
    'config': {
        'normal_p2_lateral_mode': nnk.p2_lateral_mode,
        'topdown_p2_lateral_mode': tnk.p2_lateral_mode,
        'start_level': tnk.start_level,
        'num_outs': tnk.num_outs
    }
}
Path(OUTPUT).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if report['status'] != 'PASS':
    raise SystemExit(1)
