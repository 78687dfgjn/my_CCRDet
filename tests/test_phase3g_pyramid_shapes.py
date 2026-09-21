"""Config-level and shape-contract checks for Phase 3G pyramids."""
from mmcv import Config


def test_phase3g_config_contracts():
    cases = {
        'configs_local/exp_proxy_p3p6_seed0.py': {
            'start_level': 1, 'num_outs': 4,
            'strides': [8, 16, 32, 64],
            'fusion_types': ['fusion', 'fusion', 'fusion', 'fusion_cat'],
        },
        'configs_local/exp_proxy_p2p7_seed0.py': {
            'start_level': 0, 'num_outs': 6,
            'strides': [4, 8, 16, 32, 64, 128],
            'fusion_types': [
                'fusion_cat', 'fusion', 'fusion', 'fusion',
                'fusion_cat', 'fusion_cat'],
        },
    }
    for path, expected in cases.items():
        cfg = Config.fromfile(path)
        assert cfg.model.neck.start_level == expected['start_level']
        assert cfg.model.neck.num_outs == expected['num_outs']
        assert list(cfg.model.bbox_head.anchor_generator.strides) == expected[
            'strides']
        assert list(cfg.model.fusion_types) == expected['fusion_types']
        assert cfg.model.attention_mode == 'full'
        assert cfg.model.global_shift.mode == 'off'
        assert cfg.model.p2_detail.enabled is False
        assert cfg.model.p2_alignment.enabled is False
