# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone, build_neck
from ..utils.global_shift import GlobalThermalShift
from ..utils.p2_detail_injection import P2DetailInjection
from .single_stage import SingleStageDetector
import torch
import torch.nn as nn

def winner_takes_all(tensor1, tensor2, k=5):
    diff = tensor1 - tensor2
    exp_diff = torch.exp(torch.clamp(k * diff, max=50))  
    exp_neg_diff = torch.exp(torch.clamp(k * -diff, max=50))
    w1 = exp_diff / (exp_diff + exp_neg_diff)
    w2 = 1 - w1
    return w1, w2


@DETECTORS.register_module()
class GFLAF(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 tanh=None,
                 fusion_types=None,
                 global_shift=None,
                 attention_mode=None,
                 attention_hw_threshold=4096,
                 p2_detail=None):
        super(GFLAF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        self.tanh = tanh
        # None preserves the released full-attention baseline. Candidate
        # configs opt into 'skip' explicitly for high-resolution P2 safety.
        self.attention_mode = 'full' if attention_mode is None else str(attention_mode).lower()
        if self.attention_mode not in ('full', 'skip'):
            raise ValueError('attention_mode must be full or skip')
        self.attention_hw_threshold = int(attention_hw_threshold)
        if self.attention_hw_threshold <= 0:
            raise ValueError('attention_hw_threshold must be positive')
        self.global_thermal_shift = GlobalThermalShift(
            **(global_shift or dict(mode='off')))
        p2_detail = p2_detail or {}
        self.p2_detail_injection = None
        if p2_detail.get('enabled', False):
            self.p2_detail_injection = P2DetailInjection(
                in_channels=256,
                gate_channels=p2_detail.get('gate_channels', 1),
                zero_init=p2_detail.get('zero_init', True),
                init_mode=p2_detail.get('init_mode', None))
        self.nect_t = build_neck(neck)
        self.fusion_types = list(fusion_types or
                                 ['fusion', 'fusion', 'fusion',
                                  'fusion_cat', 'fusion_cat'])
        self.fuse = nn.ModuleList([
            self._build_fusion(fusion_type) for fusion_type in self.fusion_types
        ])

    def _build_fusion(self, fusion_type):
        fusion_type = str(fusion_type).lower()
        if fusion_type == 'fusion':
            return Fusion(
                256,
                tanh=self.tanh,
                attention_mode=self.attention_mode,
                attention_hw_threshold=self.attention_hw_threshold)
        if fusion_type == 'fusion_cat':
            return Fusion_CAT(256)
        raise ValueError('Unsupported fusion type: {}'.format(fusion_type))
      
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        v_img, t_img = img
        t_img = self.global_thermal_shift(t_img)
        x, y = self.backbone(v_img, t_img)

        # The released FPN starts at backbone C3/P3.  SPDI reads C2/P2 before
        # the unchanged neck and uses it only as a detail branch.
        p2_rgb = p2_thermal = None
        if self.p2_detail_injection is not None:
            if len(x) == 0 or len(y) == 0:
                raise RuntimeError('SPDI requires backbone C2/P2 features')
            p2_rgb, p2_thermal = x[0], y[0]
        
        if self.with_neck:
            x = self.neck(x)
            y = self.nect_t(y)
      
        if len(x) != len(y) or len(x) != len(self.fuse):
            raise RuntimeError(
                'Fusion feature count mismatch: len(rgb_features)={}, '
                'len(thermal_features)={}, len(fusion_modules)={}, '
                'fusion_types={}'.format(
                    len(x), len(y), len(self.fuse), self.fusion_types))

        features = []       
        # Fusion
        for i in range(len(x)):
            feat = self.fuse[i](x[i], y[i])
            if i == 0 and self.p2_detail_injection is not None:
                feat = self.p2_detail_injection(feat, p2_rgb, p2_thermal)
            features.append(feat)
     
        return features


class Fusion_CAT(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(2 * in_channels, in_channels, 1)

    def forward(self, en_ir, en_vi):
        temp = torch.cat((en_ir, en_vi), 1)
        temp = self.conv1x1(temp)
        return temp

class Fusion_CAT_WTA(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(2 * in_channels, in_channels, 1)

    def forward(self, rgb, thermal):
        w1, w2 = winner_takes_all(rgb, thermal, k=5)
        rgb, thermal = rgb * w1, thermal * w2  
        temp = torch.cat((rgb, thermal), 1)
        temp = self.conv1x1(temp)
        return temp
    
class Fusion(nn.Module):
    def __init__(self, dim, tanh=False, attention_mode='full',
                 attention_hw_threshold=4096):
        super().__init__()
        self.dim = dim
        self.tanh = tanh
        self.attention_mode = str(attention_mode).lower()
        if self.attention_mode not in ('full', 'skip'):
            raise ValueError('attention_mode must be full or skip')
        self.attention_hw_threshold = int(attention_hw_threshold)
        if self.attention_hw_threshold <= 0:
            raise ValueError('attention_hw_threshold must be positive')
        # Analysis-only counters. They are plain Python attributes, so they do
        # not enter state_dict and remain numerically inert when disabled.
        self.analysis_enabled = False
        self.analysis_total_calls = 0
        self.analysis_skip_calls = 0
        self.analysis_full_calls = 0
        self.Q_rgb = ModalityNorm(self.dim)
        self.Q_thermal = ModalityNorm(self.dim)
        self.K_rgb = nn.Conv2d(self.dim, self.dim, 1, 1)
        self.K_thermal = nn.Conv2d(self.dim, self.dim, 1, 1)
        self.V_rgb = nn.Conv2d(self.dim, self.dim, 1, 1)
        self.V_thermal = nn.Conv2d(self.dim, self.dim, 1, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgb, thermal):   
        _, _, H, W = rgb.shape
        if self.analysis_enabled:
            self.analysis_total_calls += 1
        if (self.attention_mode == 'skip' and
                H * W > self.attention_hw_threshold):
            if self.analysis_enabled:
                self.analysis_skip_calls += 1
            return rgb + thermal
        if self.analysis_enabled:
            self.analysis_full_calls += 1
        rgb_Q = self.Q_rgb(rgb, thermal) 
        thermal_Q = self.Q_thermal(thermal, rgb)  
        
        w1, w2 = winner_takes_all(rgb_Q, thermal_Q, k=5)
        rgb_Q, thermal_Q = rgb_Q * w1, thermal_Q * w2     
        rgb_K = self.K_rgb(rgb)
        rgb_V = self.V_rgb(rgb)
        thermal_K = self.K_thermal(thermal)
        thermal_V = self.V_thermal(thermal)

        B, C, H, W = rgb_Q.shape
        q_rgb, k_rgb, v_rgb = rgb_Q.view(B,C,-1), rgb_K.view(B,C,-1), rgb_V.view(B,C,-1)
        q_thermal, k_thermal, v_thermal = thermal_Q.view(B,C,-1), thermal_K.view(B,C,-1), thermal_V.view(B,C,-1)

        attn_rgb = (q_rgb.transpose(-2, -1) @ k_rgb) / (C ** 0.5)
        attn_rgb = self.softmax(attn_rgb)
        attn_thermal = (q_thermal.transpose(-2, -1) @ k_thermal) / (C ** 0.5)
        attn_thermal = self.softmax(attn_thermal)

        rgb_out = (v_rgb @ attn_rgb).view(B, C, H, W) + rgb
        if self.tanh:
            rgb_out = torch.tanh(rgb_out)  
        thermal_out = (v_thermal @ attn_thermal).view(B, C, H, W) + thermal
        if self.tanh:
            thermal_out = torch.tanh(thermal_out)  

        out = rgb_out + thermal_out
        if self.tanh:
            out = torch.tanh(out)  
        return out
    
class ModalityNorm(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(ModalityNorm, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
                                      nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            x = self.conv(lr)
            gamma = self.conv_gamma(x)
            beta = self.conv_beta(x)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out
