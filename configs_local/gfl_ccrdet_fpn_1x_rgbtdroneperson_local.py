_base_ = [
    '../config_ccrdet/gfl_ccrdet_fpn_1x_rgbtdroneperson.py'
]

data = dict(
    train=dict(
        ann_file='/root/CCRDet/train_thermal.json',
        img_prefix='/root/CCRDet/train/'),
    val=dict(
        ann_file='/root/CCRDet/val_thermal.json',
        img_prefix='/root/CCRDet/val/'),
    test=dict(
        ann_file='/root/CCRDet/val_thermal.json',
        img_prefix='/root/CCRDet/val/'))

work_dir = '/root/CCRDet/work_dirs/gfl_ccrdet_fpn/rgbtdroneperson'
