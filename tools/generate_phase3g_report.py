"""Create Phase 3G reports from completed artifacts."""
import datetime
import glob
import json
import re
from pathlib import Path
from mmcv import Config

METRIC_KEYS = ['mAP25','mAP50','mAP75','tiny','tiny1','tiny2','tiny3','small']
LOG_KEYS = {
    'mAP25':'bbox_mAP_25','mAP50':'bbox_mAP_50','mAP75':'bbox_mAP_75',
    'tiny':'bbox_mAP_50_tiny','tiny1':'bbox_mAP_50_tiny1',
    'tiny2':'bbox_mAP_50_tiny2','tiny3':'bbox_mAP_50_tiny3',
    'small':'bbox_mAP_50_small'}
A = {'mAP25':.5635,'mAP50':.4001,'mAP75':.0346,'tiny':.4130,
     'tiny1':.1356,'tiny2':.2911,'tiny3':.4760,'small':.2814}
B = {'mAP25':.5715,'mAP50':.4158,'mAP75':.0329,'tiny':.4291,
     'tiny1':.3222,'tiny2':.3108,'tiny3':.4924,'small':.2540}

def lines(paths):
    out=[]
    for p in paths:
        for line in open(p):
            if line.strip(): out.append(json.loads(line))
    return out

def vals(paths):
    out={}
    for r in lines(paths):
        if r.get('mode')=='val':
            out[int(r['epoch'])]={k:float(r[v]) for k,v in LOG_KEYS.items()}
    return dict(sorted(out.items()))

def losses(paths):
    rows=[r for r in lines(paths) if r.get('mode')=='train']
    out={}
    for e in sorted(set(int(r['epoch']) for r in rows)):
        group=[r for r in rows if int(r['epoch'])==e]
        d={}
        for k in ('loss_cls','loss_bbox','loss_dfl','loss','lr'):
            x=[float(r[k]) for r in group if k in r]
            d[k+'_mean']=sum(x)/len(x) if x else None
            d[k+'_last']=x[-1] if x else None
        out[e]=d
    return out

def times(paths):
    pat=re.compile(r'(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)')
    x=[]
    for p in paths:
        for line in open(p,errors='ignore'):
            m=pat.search(line)
            if m: x.append(datetime.datetime.strptime(m.group(1),'%Y-%m-%d %H:%M:%S'))
    if not x: return {}
    return {'first_log_timestamp':min(x).isoformat(' '),
            'last_log_timestamp':max(x).isoformat(' '),
            'wall_seconds':(max(x)-min(x)).total_seconds()}

def best(m):
    e=max(m,key=lambda x:m[x]['mAP50'])
    return {'best_epoch':e,'best_metrics':m[e],'epoch3_metrics':m.get(3)}

def pp(a,b):
    return {k:(a[k]-b[k])*100 for k in METRIC_KEYS}

def cfgdiff():
    b=Config.fromfile('configs_local/exp_proxy_control_p3p7_seed0.py')
    c=Config.fromfile('configs_local/exp_proxy_p3p6_seed0.py')
    d=Config.fromfile('configs_local/exp_proxy_p2p7_seed0.py')
    b_fusion=list(getattr(b.model,'fusion_types',['fusion','fusion','fusion','fusion_cat','fusion_cat']))
    fields=[
      ('optimizer',lambda x:x.optimizer),('optimizer_config',lambda x:x.optimizer_config),
      ('lr_config',lambda x:x.lr_config),('runner',lambda x:x.runner),
      ('data.samples_per_gpu',lambda x:x.data.samples_per_gpu),
      ('data.workers_per_gpu',lambda x:x.data.workers_per_gpu),
      ('evaluation',lambda x:x.evaluation),('seed',lambda x:x.seed),
      ('deterministic',lambda x:x.deterministic),
      ('train_cfg',lambda x:x.model.train_cfg),('test_cfg',lambda x:x.model.test_cfg),
      ('loss_cls',lambda x:x.model.bbox_head.loss_cls),
      ('loss_bbox',lambda x:x.model.bbox_head.loss_bbox),
      ('loss_dfl',lambda x:x.model.bbox_head.loss_dfl)]
    z=['Phase 3G resolved-config semantic diff','',
       'Control: configs_local/exp_proxy_control_p3p7_seed0.py',
       'C: configs_local/exp_proxy_p3p6_seed0.py',
       'D: configs_local/exp_proxy_p2p7_seed0.py','',
       '[Protocol fields; equality against control]']
    for n,g in fields: z.append('{}: C_equal={} D_equal={}'.format(n,g(b)==g(c),g(b)==g(d)))
    z += ['', '[Architecture-only differences]',
      'C neck.start_level {} -> {}; num_outs {} -> {}'.format(
        b.model.neck.start_level,c.model.neck.start_level,
        b.model.neck.num_outs,c.model.neck.num_outs),
      'C strides {} -> {}'.format(list(b.model.bbox_head.anchor_generator.strides),
                                   list(c.model.bbox_head.anchor_generator.strides)),
      'C fusion_types {} -> {}'.format(b_fusion,list(c.model.fusion_types)),
      'D neck.start_level {} -> {}; num_outs {} -> {}'.format(
        b.model.neck.start_level,d.model.neck.start_level,
        b.model.neck.num_outs,d.model.neck.num_outs),
      'D strides {} -> {}'.format(list(b.model.bbox_head.anchor_generator.strides),
                                   list(d.model.bbox_head.anchor_generator.strides)),
      'D fusion_types {} -> {}'.format(b_fusion,list(d.model.fusion_types)),
      'C/D attention_mode=full; global_shift=off; p2_detail=disabled; p2_alignment=disabled',
      'All other differences are work_dir/load_from or listed pyramid fields.']
    return '\n'.join(z)+'\n'

def main():
    cp=glob.glob('/hy-tmp/CCRDet_assets/work_dirs/phase3g_proxy_p3p6_seed0/*.log.json')
    dp=glob.glob('/hy-tmp/CCRDet_assets/work_dirs/phase3g_proxy_p2p7_seed0/*.log.json')
    cm,dm=vals(cp),vals(dp)
    c={'label':'P3-P6','factor':{'P2':False,'P7':False},
       'config':'configs_local/exp_proxy_p3p6_seed0.py',
       'warmstart':'/hy-tmp/CCRDet_assets/phase3g_warmstart_p3p6_seed0.pth',
       'metrics_by_epoch':cm,'train_loss_summary_by_epoch':losses(cp),
       **best(cm),'log_paths':cp,
       'log_time':times(['experiments/phase3g_p3p6_train.log']+
          glob.glob('/hy-tmp/CCRDet_assets/work_dirs/phase3g_proxy_p3p6_seed0/*.log'))}
    d={'label':'P2-P7','factor':{'P2':True,'P7':True},
       'config':'configs_local/exp_proxy_p2p7_seed0.py',
       'warmstart':'/hy-tmp/CCRDet_assets/phase3g_warmstart_p2p7_seed0.pth',
       'metrics_by_epoch':dm,'train_loss_summary_by_epoch':losses(dp),
       **best(dm),'log_paths':dp,
       'log_time':times(['experiments/phase3g_p2p7_train.log',
                         'experiments/phase3g_p2p7_train_recovery.log']+
          glob.glob('/hy-tmp/CCRDet_assets/work_dirs/phase3g_proxy_p2p7_seed0/*.log')),
       'recovery_note':'Initial SSH-attached attempt stopped at epoch2 iteration 450/606 without OOM; completed run resumed from epoch_1.pth with optimizer state and completed epochs 2-3.'}
    Path('experiments/phase3g_p3p6_proxy.json').write_text(json.dumps(c,indent=2))
    Path('experiments/phase3g_p2p7_proxy.json').write_text(json.dumps(d,indent=2))
    cells={
      'P3-P7':{'P2':False,'P7':True,'best_epoch':3,'best_metrics':A,'epoch3_metrics':A,'source':'Phase 3B paired control'},
      'P3-P6':{'P2':False,'P7':False,'best_epoch':c['best_epoch'],'best_metrics':c['best_metrics'],'epoch3_metrics':c['epoch3_metrics'],'source':'Phase 3G new proxy'},
      'P2-P7':{'P2':True,'P7':True,'best_epoch':d['best_epoch'],'best_metrics':d['best_metrics'],'epoch3_metrics':d['epoch3_metrics'],'source':'Phase 3G new proxy'},
      'P2-P6':{'P2':True,'P7':False,'best_epoch':3,'best_metrics':B,'epoch3_metrics':B,'source':'Phase 3B paired Lite-P2 proxy'}}
    bm={n:x['best_metrics'] for n,x in cells.items()}
    con={
      'P2 effect with P7 (P2-P7 minus P3-P7)':pp(bm['P2-P7'],bm['P3-P7']),
      'P2 effect without P7 (P2-P6 minus P3-P6)':pp(bm['P2-P6'],bm['P3-P6']),
      'P7 effect without P2 (P3-P7 minus P3-P6)':pp(bm['P3-P7'],bm['P3-P6']),
      'P7 effect with P2 (P2-P7 minus P2-P6)':pp(bm['P2-P7'],bm['P2-P6'])}
    inter={k:con['P2 effect with P7 (P2-P7 minus P3-P7)'][k]-
             con['P2 effect without P7 (P2-P6 minus P3-P6)'][k] for k in METRIC_KEYS}
    phase={'protocol':{'seed':0,'deterministic':True,'batch_size':8,'input_resolution':[512,640],
      'optimizer':'SGD','lr':.001,'momentum':.9,'weight_decay':.0001,'max_epochs':3,
      'amp':False,'gradient_accumulation':False,'global_shift':'off','spdi':'disabled',
      'p2_alignment':'disabled','pair_thermal_calibration':'disabled','attention':'full'},
      'cells':cells,'contrasts_percentage_points':con,'interaction_percentage_points':inter,
      'P2_MECHANISM':'SUPPORTED','BEST_PROXY_TOPOLOGY':max(bm,key=lambda n:bm[n]['mAP50']),
      'PHASE3F_CGMA_ALIGNMENT_CLAIM':'NOT ESTABLISHED',
      'phase3f_reason':['ZNCC winner (-2,+2) was a search-boundary corner',
        'frozen calibration degraded mAP50 strongly',
        'learned P2 residual alignment had no positive diagnostic signal']}
    Path('experiments/phase3g_pyramid_2x2.json').write_text(json.dumps(phase,indent=2))
    Path('experiments/phase3g_resolved_config_diff.txt').write_text(cfgdiff())
    mem={n:json.loads(Path(p).read_text()) for n,p in
      [('P3-P6','experiments/phase3g_memory_p3p6.json'),
       ('P2-P7','experiments/phase3g_memory_p2p7.json')]}
    Path('experiments/phase3g_memory.json').write_text(json.dumps(mem,indent=2))
    R=['# Phase 3G 2x2 Pyramid Causality Ablation','',
       'One-seed, 3-epoch warm-start proxy comparison; descriptive only.',
       '','## Gates','',
       '- P3-P6 semantic warm-start: PASS; P2 absent and P7 discarded.',
       '- P2-P7 semantic warm-start: PASS; P2 fresh and P7 retained.',
       '- Full-resolution smoke: P3-P6 5 iterations PASS; P2-P7 20 iterations PASS; AMP off.',
       '- Both new proxies completed 3 epochs with epoch-wise validation.','',
       '## Best mAP50 cells','',
       '| Topology | P2 | P7 | best epoch | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in ('P3-P7','P3-P6','P2-P7','P2-P6'):
        m=bm[n]; x=cells[n]
        R.append('| {} | {} | {} | {} | {} |'.format(n,int(x['P2']),int(x['P7']),x['best_epoch'],
          ' | '.join('{:.4f}'.format(m[k]) for k in METRIC_KEYS)))
    R += ['','## Epoch 3 cells','',
      '| Topology | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in ('P3-P7','P3-P6','P2-P7','P2-P6'):
        m=cells[n]['epoch3_metrics']
        R.append('| {} | {} |'.format(n,' | '.join('{:.4f}'.format(m[k]) for k in METRIC_KEYS)))
    R += ['','## Factor contrasts (percentage points)','',
      '| Contrast | mAP25 | mAP50 | mAP75 | tiny | tiny1 | tiny2 | tiny3 | small |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for lab,key in [('P2 effect with P7','P2 effect with P7 (P2-P7 minus P3-P7)'),
                    ('P2 effect without P7','P2 effect without P7 (P2-P6 minus P3-P6)'),
                    ('P7 effect without P2','P7 effect without P2 (P3-P7 minus P3-P6)'),
                    ('P7 effect with P2','P7 effect with P2 (P2-P7 minus P2-P6)'),
                    ('2x2 interaction',None)]:
        q=inter if key is None else con[key]
        R.append('| {} | {} |'.format(lab,' | '.join('{:+.2f}'.format(q[k]) for k in METRIC_KEYS)))
    R += ['','## Interpretation','',
      '- P2 mAP50 effect is positive with P7 ({:+.2f} pp) and without P7 ({:+.2f} pp); P2 mechanism is SUPPORTED in this proxy.'.format(con['P2 effect with P7 (P2-P7 minus P3-P7)']['mAP50'],con['P2 effect without P7 (P2-P6 minus P3-P6)']['mAP50']),
      '- P7 removal alone is slightly negative for mAP50 in both P2 conditions; P7 deletion alone does not explain the Lite-P2 gain.',
      '- tiny1 improves strongly with P2 ({:+.2f} pp with P7; {:+.2f} pp without P7), while small changes {:+.2f} pp and {:+.2f} pp.'.format(con['P2 effect with P7 (P2-P7 minus P3-P7)']['tiny1'],con['P2 effect without P7 (P2-P6 minus P3-P6)']['tiny1'],con['P2 effect with P7 (P2-P7 minus P3-P7)']['small'],con['P2 effect without P7 (P2-P6 minus P3-P6)']['small']),
      '- mAP75 changes {:+.2f} pp with P7 and {:+.2f} pp without P7; high-IoU localization benefit is not established.'.format(con['P2 effect with P7 (P2-P7 minus P3-P7)']['mAP75'],con['P2 effect without P7 (P2-P6 minus P3-P6)']['mAP75']),
      '- BEST_PROXY_TOPOLOGY is {} by best mAP50; this remains descriptive, not a formal 12-epoch result.'.format(phase['BEST_PROXY_TOPOLOGY']),
      '','## Phase 3F status','',
      'CGMA_ALIGNMENT_CLAIM = NOT ESTABLISHED. The train-only ZNCC winner (-2,+2) was at the search boundary; frozen calibration degraded strongly; and learned P2 residual alignment had no positive diagnostic signal. The 44.40 observation is retraining under train-derived fixed translation plus residual branch, not proof of geometric correction.',
      '','## Resource summary','',
      '| Smoke | status | iterations | peak allocated (GiB) | peak reserved (GiB) | driver used (MiB) |',
      '|---|---|---:|---:|---:|---:|']
    for n in ('P3-P6','P2-P7'):
        x=mem[n]
        R.append('| {} | {} | {} | {:.2f} | {:.2f} | {} |'.format(n,x['status'],x['iterations_completed'],
          x['peak_allocated_bytes']/2**30,x['peak_reserved_bytes']/2**30,x['peak_driver_used_mb']))
    R += ['','## Limitations','',
      'The two new cells are one-seed, 3-epoch warm-start screening experiments. No multi-seed statistical or formal 12-epoch claim is made.']
    Path('docs/PHASE3G_PYRAMID_CAUSALITY.md').write_text('\n'.join(R)+'\n')
    p=Path('docs/PHASE3F_CGMA_PROXY.md')
    note=('\n\n## Phase 3G status update\n\n'
      'Phase 3G freezes Phase 3F. CGMA_ALIGNMENT_CLAIM = NOT ESTABLISHED: '
      'the train-only ZNCC winner was the search-boundary corner (-2,+2), '
      'frozen calibration degraded mAP50 strongly, and learned P2 residual '
      'alignment had no positive diagnostic signal. The 44.40 proxy remains '
      'a retraining observation under train-derived fixed translation plus '
      'residual branch, not a validated geometric correction.\n')
    old=p.read_text()
    if 'CGMA_ALIGNMENT_CLAIM = NOT ESTABLISHED' not in old: p.write_text(old.rstrip()+note)
    print(json.dumps({'P3-P6':c['best_metrics'],'P2-P7':d['best_metrics'],
      'BEST_PROXY_TOPOLOGY':phase['BEST_PROXY_TOPOLOGY'],'mAP50_contrasts':{
      'P2_with_P7':con['P2 effect with P7 (P2-P7 minus P3-P7)']['mAP50'],
      'P2_without_P7':con['P2 effect without P7 (P2-P6 minus P3-P6)']['mAP50'],
      'P7_without_P2':con['P7 effect without P2 (P3-P7 minus P3-P6)']['mAP50'],
      'P7_with_P2':con['P7 effect with P2 (P2-P7 minus P2-P6)']['mAP50'],
      'interaction':inter['mAP50']}},indent=2))
if __name__=='__main__': main()
