"""
W-KANet: Wavelet-Guided Kolmogorov-Arnold Network with
         Tri-Stream Boundary-Reverse Fusion for Cardiac Segmentation
============================================================================

Official Implementation of:
"W-KANet: A Wavelet-Guided Kolmogorov-Arnold Network for Robust 
Cardiac Segmentation in Echocardiographic Images"

Architecture Components:
  1. MSFE        — Multi-Scale Feature Encoder (RSDBlock, stages 1-3)
  2. WKGMSABlock — Wavelet-guided KAN + Grouped Multi-Scale Attention
  3. TSBRF       — Tri-Stream Boundary-Reverse Fusion:
      - DAFM (Dual ASPP Fusion Module)
      - RAM  (Reverse Attention Module)
      - MBD  (Multi-scale Boundary Detector with Attention)
  4. HDRD        — Hierarchical Dense Residual Decoder

Repository: https://github.com/Hassan48khan/W-KANet
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Helper Modules
# ═══════════════════════════════════════════════════════════════════════════

def to_2tuple(x):
    """Convert scalar or tuple to 2-tuple."""
    return (int(x), int(x)) if isinstance(x, (int, float)) else x


class DropPath(nn.Module):
    """Stochastic Depth (drop_path) regularization."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        r = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        r.floor_()
        return x / keep_prob * r


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network Linear Layer.
    
    Implements learnable activation functions via B-splines, enabling
    explicit modeling of complex nonlinear transformations.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        grid_size: Number of B-spline grid points (default: 5)
        spline_order: Order of B-spline (default: 3)
        scale_noise: Initial noise scale (default: 0.1)
        scale_base: Base activation scale (default: 1.0)
        scale_spline: Spline activation scale (default: 1.0)
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True,
                 base_activation=nn.SiLU, grid_eps=0.02, grid_range=(-1, 1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                     * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """Compute B-spline basis functions."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (((x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * bases[:, :, :-1])
                   + ((grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]))
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """Convert curve to B-spline coefficients via least squares."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        return torch.linalg.lstsq(A, B).solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(self.b_splines(x).view(x.size(0), -1),
                              self.scaled_spline_weight.view(self.out_features, -1))
        return base_out + spline_out

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """KAN regularization loss for sparsity and entropy."""
        l1 = self.spline_weight.abs().mean(-1)
        s = l1.sum()
        p = l1 / (s + 1e-8)
        return regularize_activation * s + regularize_entropy * (-torch.sum(p * p.log()))


class DW_bn_relu(nn.Module):
    """Depthwise convolution + BatchNorm + ReLU."""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        return self.relu(self.bn(self.dwconv(x))).flatten(2).transpose(1, 2)


class PatchEmbed(nn.Module):
    """Patch embedding layer for tokenization."""
    def __init__(self, img_size=224, patch_size=7, stride=4,
                 in_chans=3, embed_dim=768):
        super().__init__()
        ps = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=ps, stride=stride,
                              padding=(ps[0] // 2, ps[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — MSFE: Multi-Scale Feature Encoder (RSDBlock)
# ═══════════════════════════════════════════════════════════════════════════

class RSDBlock(nn.Module):
    """
    Residual Squeeze-Dilated Block.
    
    Combines:
    - Cascaded dilated convolutions for multi-scale receptive fields
    - Squeeze-and-Excitation (SE) channel attention
    - Residual shortcut connections
    
    Used in:
    - Encoder stages 1-3 (MSFE)
    - Decoder stages 1-5 (HDRD)
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        dilation_rates: Dilation rates for cascaded convolutions
        reduction: SE reduction ratio
    """
    def __init__(self, in_ch, out_ch, dilation_rates=(1, 2, 4), reduction=8):
        super().__init__()
        assert len(dilation_rates) >= 2, "Need at least 2 dilation rates"
        mid_ch = max(in_ch, out_ch)

        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3,
                               padding=dilation_rates[0],
                               dilation=dilation_rates[0], bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3,
                               padding=dilation_rates[1],
                               dilation=dilation_rates[1], bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if len(dilation_rates) >= 3:
            self.conv3 = nn.Conv2d(out_ch, out_ch, 3,
                                   padding=dilation_rates[2],
                                   dilation=dilation_rates[2], bias=False)
            self.bn3 = nn.BatchNorm2d(out_ch)
        else:
            self.conv3 = None

        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, max(1, out_ch // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, out_ch // reduction), out_ch, 1),
            nn.Sigmoid()
        )

        self.shortcut = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                       nn.BatchNorm2d(out_ch))
                         if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        if self.conv3 is not None:
            out = self.bn3(self.conv3(out))
        out = out * self.se(out)
        return self.relu(out + identity)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — WKGMSABlock: Wavelet-guided KAN + Grouped Multi-Scale Attention
# ═══════════════════════════════════════════════════════════════════════════

class _HaarFn(torch.autograd.Function):
    """Haar Discrete Wavelet Transform with custom backward pass."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ll = (x[:,:,0::2,0::2]+x[:,:,0::2,1::2]+x[:,:,1::2,0::2]+x[:,:,1::2,1::2]) * 0.25
        lh = (x[:,:,0::2,0::2]-x[:,:,0::2,1::2]+x[:,:,1::2,0::2]-x[:,:,1::2,1::2]) * 0.25
        hl = (x[:,:,0::2,0::2]+x[:,:,0::2,1::2]-x[:,:,1::2,0::2]-x[:,:,1::2,1::2]) * 0.25
        hh = (x[:,:,0::2,0::2]-x[:,:,0::2,1::2]-x[:,:,1::2,0::2]+x[:,:,1::2,1::2]) * 0.25
        return ll, lh, hl, hh

    @staticmethod
    def backward(ctx, g_ll, g_lh, g_hl, g_hh):
        x, = ctx.saved_tensors
        g = torch.zeros_like(x)
        g[:,:,0::2,0::2] = g_ll+g_lh+g_hl+g_hh
        g[:,:,0::2,1::2] = g_ll-g_lh+g_hl-g_hh
        g[:,:,1::2,0::2] = g_ll+g_lh-g_hl-g_hh
        g[:,:,1::2,1::2] = g_ll-g_lh-g_hl+g_hh
        return g


class HaarDWT2D(nn.Module):
    """2D Haar Discrete Wavelet Transform."""
    def forward(self, x: torch.Tensor):
        ph = x.shape[2] % 2
        pw = x.shape[3] % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return _HaarFn.apply(x)


class WaveletKANLayer(nn.Module):
    """
    Wavelet-guided KAN Layer.
    
    Frequency-Domain Guidance:
    - LL subband (low-freq) → guides KAN base activation (SiLU) for semantics
    - HH+LH+HL subbands (high-freq) → guides B-spline path for edge details
    
    Args:
        dim: Feature dimension
    """
    def __init__(self, dim: int):
        super().__init__()
        self.kan = KANLinear(dim, dim)
        self.dw = DW_bn_relu(dim)
        self.norm = nn.LayerNorm(dim)
        self.low_proj = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.Sigmoid())
        self.high_proj = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.Sigmoid())

    def forward(self, x, ll_up, hh_up, H, W):
        B, N, C = x.shape
        lb = self.low_proj(ll_up).flatten(2).transpose(1, 2)
        hb = self.high_proj(hh_up).flatten(2).transpose(1, 2)
        xf = x.reshape(B*N, C)
        lb = lb.reshape(B*N, C)
        hb = hb.reshape(B*N, C)
        base_out = F.linear(self.kan.base_activation(xf + lb), self.kan.base_weight)
        spline_out = F.linear(self.kan.b_splines(xf + hb).view(B*N, -1),
                              self.kan.scaled_spline_weight.view(self.kan.out_features, -1))
        z = (base_out + spline_out).reshape(B, N, C)
        z = self.dw(z, H, W)
        return self.norm(z) + x


class WKGMSABlock(nn.Module):
    """
    Wavelet-guided KAN with Grouped Multi-Scale Attention Block.
    
    Two-Stage Processing:
    
    Stage 1 - Wavelet-guided KAN:
      - Haar DWT decomposes features into LL/HH subbands
      - LL guides base (SiLU) activation path
      - HH guides spline (B-spline) activation path
    
    Stage 2 - Grouped Multi-Scale Attention:
      - G groups for parallel processing (default: 32)
      - Branch A: 1-D horizontal/vertical AvgPool
      - Branch C: 3×3 conv + dual-pool fusion
      - Cross-branch matmul for attention weights
      - Linear O(Nd²) computational complexity
    
    Args:
        dim: Feature dimension
        groups: Number of attention groups (default: 32)
        drop_path: Drop path rate (default: 0)
        kan_depth: Number of KAN layers (default: 2)
    """
    def __init__(self, dim: int, groups: int = 32, drop_path: float = 0.,
                 kan_depth: int = 2, norm_layer=nn.LayerNorm):
        super().__init__()
        # Auto-adjust groups for divisibility
        while dim % groups != 0 and groups > 1:
            groups //= 2
        self.G = groups
        self.Cg = dim // groups

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.dwt = HaarDWT2D()
        self.alpha_ll = nn.Parameter(torch.tensor(1.0))
        self.alpha_hh = nn.Parameter(torch.tensor(1.0))
        self.token_proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.token_norm = norm_layer(dim)
        self.wk_layers = nn.ModuleList([WaveletKANLayer(dim) for _ in range(kan_depth)])

        # Grouped Multi-Scale Attention
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1_A = nn.Conv2d(self.Cg, self.Cg, 1, bias=False)
        self.gn = nn.GroupNorm(1, self.Cg)
        self.conv3x3_C = nn.Conv2d(self.Cg, self.Cg, 3, 1, 1, bias=False)
        self.avgp_C = nn.AdaptiveAvgPool2d((1, 1))
        self.maxp_C = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_C = nn.Conv2d(self.Cg * 2, self.Cg, 1, bias=False)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(-1)

    @staticmethod
    def _up(sub, target):
        if sub.shape[-2:] == target.shape[-2:]:
            return sub
        return F.interpolate(sub, target.shape[-2:], mode='bilinear', align_corners=False)

    def _wavelet_kan(self, x_tokens, H, W):
        """Apply wavelet-guided KAN transformation."""
        B, N, C = x_tokens.shape
        zi = self.token_norm(x_tokens).transpose(1, 2).reshape(B, C, H, W)
        zi = self.token_proj(zi)
        ll, lh, hl, hh = self.dwt(zi)
        ll_up = self.alpha_ll * self._up(ll, zi)
        hh_up = self.alpha_hh * (self._up(hh, zi) + self._up(lh, zi) + self._up(hl, zi))
        z = zi.flatten(2).transpose(1, 2)
        for layer in self.wk_layers:
            z = layer(z, ll_up, hh_up, H, W)
        return z

    def _grouped_attn(self, img):
        """Apply grouped multi-scale attention."""
        B, C, H, W = img.shape
        gx = img.reshape(B * self.G, self.Cg, H, W)
        xh = self.pool_h(gx)
        xw = self.pool_w(gx).permute(0, 1, 3, 2)
        hw = self.conv1x1_A(torch.cat([xh, xw], dim=2))
        xh2, xw2 = torch.split(hw, [H, W], dim=2)
        x1 = self.gn(gx * xh2.sigmoid() * xw2.permute(0, 1, 3, 2).sigmoid())
        c = self.conv3x3_C(gx)
        x2 = self.conv1x1_C(torch.cat([self.avgp_C(c), self.maxp_C(c)], dim=1)).sigmoid() * c
        x11 = self.softmax(self.agp(x1).reshape(B*self.G, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(B*self.G, self.Cg, -1)
        x21 = self.softmax(self.agp(x2).reshape(B*self.G, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(B*self.G, self.Cg, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(B*self.G, 1, H, W)
        return (gx * weights.sigmoid()).reshape(B, C, H, W)

    def forward(self, x, H, W):
        B, N, C = x.shape
        z_wk = x + self.drop_path(self._wavelet_kan(x, H, W))
        img = z_wk.transpose(1, 2).reshape(B, C, H, W)
        y_attn = self._grouped_attn(img)
        return (y_attn + img).flatten(2).transpose(1, 2)

    def regularization_loss(self, ra=1.0, re=1.0):
        """KAN regularization loss."""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for l in self.wk_layers:
            loss = loss + l.kan.regularization_loss(ra, re)
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — TSBRF: Tri-Stream Boundary-Reverse Fusion
# ═══════════════════════════════════════════════════════════════════════════

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    
    Multi-scale context aggregation using parallel dilated convolutions
    with rates {1, 6, 12, 18} and global average pooling.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        def blk(k, d):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.c2 = blk(3, 6)
        self.c3 = blk(3, 12)
        self.c4 = blk(3, 18)
        self.gp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x5 = F.interpolate(self.gp(x), size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        return self.fusion(torch.cat([self.c1(x), self.c2(x),
                                      self.c3(x), self.c4(x), x5], dim=1))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequential channel and spatial attention for feature refinement.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(1, channels // reduction)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ca(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sa(torch.cat([avg, mx], dim=1))


class DAFM(nn.Module):
    """
    Dual ASPP Fusion Module (TSBRF Stream 1).
    
    Balances encoder semantics and decoder spatial information through:
    - Dual ASPP for multi-scale context
    - Cross-multiplication for complementary feature interactions
    - Spatial attention refinement
    
    Args:
        enc_ch: Encoder feature channels
        dec_ch: Decoder feature channels
        out_ch: Output channels
    """
    def __init__(self, enc_ch: int, dec_ch: int, out_ch: int):
        super().__init__()
        self.aspp_enc = ASPP(enc_ch, out_ch)
        self.aspp_dec = ASPP(dec_ch, out_ch)
        
        # Cross-multiplication branch
        self.cross_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        
        # Concatenation fusion
        self.concat_conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        
        # Spatial attention
        self.sa_avgpool = nn.AdaptiveAvgPool2d(1)
        self.sa_maxpool = nn.AdaptiveMaxPool2d(1)
        self.sa_proj = nn.Sequential(
            nn.Conv2d(out_ch, max(1, out_ch // 4), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(1, out_ch // 4), out_ch, 1))
        self.sa_conv7 = nn.Conv2d(out_ch, 1, 7, padding=3)
    
    def forward(self, enc_feat, dec_feat):
        ae = self.aspp_enc(enc_feat)
        ad = self.aspp_dec(dec_feat)
        cross = ae * ad
        E1 = self.cross_conv(cross) + ae
        D1 = self.cross_conv(cross) + ad
        Fp = self.concat_conv(torch.cat([E1, D1], dim=1))
        # Spatial attention
        avg = self.sa_proj(self.sa_avgpool(Fp))
        mx = self.sa_proj(self.sa_maxpool(Fp))
        sa = torch.sigmoid(self.sa_conv7(avg + mx))
        return Fp * sa


class RAM(nn.Module):
    """
    Reverse Attention Module (TSBRF Stream 2).
    
    Progressive background modeling through reverse attention:
        S^r_{i+1} = 1 - sigmoid(S_{i+1})
    
    Suppresses false positive background activations through decoder stages.
    
    Args:
        out_ch: Output feature channels
    """
    def __init__(self, out_ch: int):
        super().__init__()
        self.reverse_proj = nn.Sequential(
            nn.Conv2d(1, out_ch, 1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    
    def forward(self, upper_pred, target_size):
        if upper_pred is not None:
            if upper_pred.shape[2:] != target_size:
                upper_pred = F.interpolate(upper_pred, target_size,
                                          mode='bilinear', align_corners=True)
            reverse = 1.0 - torch.sigmoid(upper_pred)
        else:
            reverse = torch.zeros(upper_pred.shape[0] if upper_pred is not None else 1, 
                                 1, target_size[0], target_size[1],
                                 device=upper_pred.device if upper_pred is not None else 'cuda')
        return self.reverse_proj(reverse)


class MBD(nn.Module):
    """
    Multi-scale Boundary Detector with Attention (TSBRF Stream 3).
    
    Architecture:
    - Multi-scale convolutions: 3×3, 5×5, 7×7 (parallel)
    - Concatenation-based fusion
    - Spatial attention: Conv7×7 → Sigmoid
    - Channel attention: GAP → MLP → Sigmoid
    - Final: (fused × spatial_att) + (fused × channel_att)
    
    Provides explicit boundary features with Canny supervision.
    
    Args:
        dec_ch: Decoder feature channels
        out_ch: Output channels
    """
    def __init__(self, dec_ch: int, out_ch: int):
        super().__init__()
        # Multi-scale convolution branches
        q = max(1, out_ch // 4)
        self.bc3 = nn.Conv2d(dec_ch, q, 3, padding=1)
        self.bc5 = nn.Conv2d(dec_ch, q, 5, padding=2)
        self.bc7 = nn.Conv2d(dec_ch, q, 7, padding=3)
        
        # Concatenation fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(q * 3, out_ch, 1),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(inplace=True))
        
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_ch, 1, 7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        
        # Channel attention
        reduction = 16
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, max(1, out_ch // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, out_ch // reduction), out_ch, 1),
            nn.Sigmoid())
    
    def forward(self, dec_feat):
        # Multi-scale boundary extraction
        b3 = self.bc3(dec_feat)
        b5 = self.bc5(dec_feat)
        b7 = self.bc7(dec_feat)
        
        # Concatenation fusion
        fused = self.fusion(torch.cat([b3, b5, b7], dim=1))
        
        # Parallel attention refinement
        spatial_feat = fused * self.spatial_attn(fused)
        channel_feat = fused * self.channel_attn(fused)
        
        # Additive fusion
        return spatial_feat + channel_feat


class TSBRF(nn.Module):
    """
    Tri-Stream Boundary-Reverse Fusion.
    
    Integrates three complementary information streams:
    
    Stream 1 (DAFM): Dual ASPP Fusion Module
        - Balances encoder semantics + decoder spatial info
    
    Stream 2 (RAM): Reverse Attention Module
        - Background modeling via reverse features
    
    Stream 3 (MBD): Multi-scale Boundary Detector with Attention
        - Multi-scale boundary extraction with dual attention
        - Canny supervision (low=50, high=150)
    
    Fusion: Concat → Attention Gating → CBAM → Residual
    
    Outputs:
        - F_i: Refined features for next decoder stage
        - S_i: Segmentation prediction
        - B_i: Boundary map
    
    Args:
        enc_ch: Encoder feature channels
        dec_ch: Decoder feature channels
        out_ch: Output channels (default: dec_ch)
    """
    def __init__(self, enc_ch: int, dec_ch: int, out_ch: int = None):
        super().__init__()
        if out_ch is None:
            out_ch = dec_ch
        self.out_ch = out_ch

        # Three streams
        self.dafm = DAFM(enc_ch, dec_ch, out_ch)
        self.ram = RAM(out_ch)
        self.mbd = MBD(dec_ch, out_ch)

        # 3-Stream Fusion
        self.stream_fusion = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.attn_gate = nn.Sequential(
            nn.Conv2d(out_ch, max(1, out_ch // 4), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(1, out_ch // 4), 1, 1), nn.Sigmoid())
        self.cbam = CBAM(out_ch)
        self.residual_proj = (nn.Sequential(nn.Conv2d(dec_ch, out_ch, 1),
                                             nn.BatchNorm2d(out_ch))
                               if dec_ch != out_ch else nn.Identity())

        # Prediction heads
        self.seg_head = nn.Conv2d(out_ch, 1, 1)
        self.boundary_head = nn.Conv2d(out_ch, 1, 1)

    def forward(self, enc_feat, dec_feat, upper_pred=None):
        # Align spatial dimensions
        if enc_feat.shape[2:] != dec_feat.shape[2:]:
            enc_feat = F.interpolate(enc_feat, dec_feat.shape[2:],
                                     mode='bilinear', align_corners=True)

        # Three streams
        s1 = self.dafm(enc_feat, dec_feat)
        s2 = self.ram(upper_pred, dec_feat.shape[2:])
        s3 = self.mbd(dec_feat)

        # 3-Stream fusion
        fused = self.stream_fusion(torch.cat([s1, s2, s3], dim=1))
        gate = self.attn_gate(fused)
        refined = self.cbam(fused * gate + self.residual_proj(dec_feat))

        return refined, self.seg_head(refined), self.boundary_head(s3)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — W-KANet: Complete Architecture
# ═══════════════════════════════════════════════════════════════════════════

class WKANet(nn.Module):
    """
    W-KANet: Wavelet-Guided Kolmogorov-Arnold Network with
             Tri-Stream Boundary-Reverse Fusion for Cardiac Segmentation.
    
    Architecture:
        Encoder (5 stages):
            - Stages 1-3: MSFE (RSDBlock) - 32 → 64 → 128 channels
            - Stages 4-5: WKGMSABlock - 320 → 512 channels
        
        Skip Connections (4 stages):
            - TSBRF (DAFM + RAM + MBD) at each decoder level
        
        Decoder (5 stages):
            - HDRD (RSDBlock) - 512 → 320 → 128 → 64 → 32 channels
    
    Args:
        num_classes: Output classes (1 for binary, 4 for multi-class)
        input_channels: Input image channels (1 for grayscale)
        img_size: Input image size (default: 256)
        embed_dims: Bottleneck channel dimensions (default: (320, 512))
        gt_ds: Enable deep supervision (default: True)
        drop_path_rate: Stochastic depth rate (default: 0.1)
        depths: Number of WKGMSABlocks per stage (default: (1, 1, 1))
    """
    def __init__(self, num_classes: int = 1, input_channels: int = 1,
                 img_size: int = 256, embed_dims=(320, 512),
                 gt_ds: bool = True, drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer=nn.LayerNorm, depths=(1, 1, 1)):
        super().__init__()
        self.gt_ds = gt_ds
        D0, D1 = embed_dims

        # MSFE: Encoder stages 1-3
        self.encoder1 = RSDBlock(input_channels, 32, dilation_rates=(1, 1))
        self.encoder2 = RSDBlock(32, 64, dilation_rates=(1, 2))
        self.encoder3 = RSDBlock(64, 128, dilation_rates=(1, 2, 3))

        # WKGMSABlock: Bottleneck stages 4-5
        self.norm3 = norm_layer(D0)
        self.norm4 = norm_layer(D1)
        self.dnorm3 = norm_layer(D0)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([WKGMSABlock(D0, drop_path=dpr[i])
                                      for i in range(depths[0])])
        self.block2 = nn.ModuleList([WKGMSABlock(D1, drop_path=dpr[sum(depths[:1])+i])
                                      for i in range(depths[1])])
        self.dblock1 = nn.ModuleList([WKGMSABlock(D0, drop_path=dpr[sum(depths[:2])+i])
                                      for i in range(depths[2])])
        self.dblock2 = nn.ModuleList([WKGMSABlock(128, drop_path=dpr[i])
                                      for i in range(depths[0])])

        self.patch_embed3 = PatchEmbed(img_size=img_size//4, patch_size=3,
                                       stride=2, in_chans=128, embed_dim=D0)
        self.patch_embed4 = PatchEmbed(img_size=img_size//8, patch_size=3,
                                       stride=2, in_chans=D0, embed_dim=D1)

        # HDRD: Decoder stages 1-5
        self.decoder1 = RSDBlock(D1, D0, dilation_rates=(1, 2, 4))
        self.decoder2 = RSDBlock(D0, 128, dilation_rates=(1, 2, 4))
        self.decoder3 = RSDBlock(128, 64, dilation_rates=(1, 2, 4))
        self.decoder4 = RSDBlock(64, 32, dilation_rates=(1, 2))
        self.decoder5 = RSDBlock(32, 32, dilation_rates=(1, 1))

        # TSBRF skip connections
        self.tsbrf4 = TSBRF(enc_ch=D0, dec_ch=D0, out_ch=D0)
        self.tsbrf3 = TSBRF(enc_ch=128, dec_ch=128, out_ch=128)
        self.tsbrf2 = TSBRF(enc_ch=64, dec_ch=64, out_ch=64)
        self.tsbrf1 = TSBRF(enc_ch=32, dec_ch=32, out_ch=32)

        # Output
        self.final_conv = nn.Conv2d(32, num_classes, 1)

        # Deep supervision
        if gt_ds:
            self.ds1 = nn.Conv2d(D0, num_classes, 1)
            self.ds2 = nn.Conv2d(128, num_classes, 1)
            self.ds3 = nn.Conv2d(64, num_classes, 1)
            self.ds4 = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]
        B = x.shape[0]

        # Encoder
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2)); e1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2)); e2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2)); e3 = out

        # Bottleneck stage 4
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e4 = out

        # Bottleneck stage 5
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Decoder stage 1 with TSBRF
        out = F.relu(F.interpolate(self.decoder1(out),
                                   scale_factor=2, mode='bilinear', align_corners=False))
        t4_feat, t4_seg, t4_bnd = self.tsbrf4(e4, out, upper_pred=None)
        out = out + t4_feat

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Decoder stage 2 with TSBRF
        out = F.relu(F.interpolate(self.decoder2(out),
                                   scale_factor=2, mode='bilinear', align_corners=False))
        t3_feat, t3_seg, t3_bnd = self.tsbrf3(e3, out, upper_pred=t4_seg)
        out = out + t3_feat

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Decoder stage 3 with TSBRF
        out = F.relu(F.interpolate(self.decoder3(out),
                                   scale_factor=2, mode='bilinear', align_corners=False))
        t2_feat, t2_seg, t2_bnd = self.tsbrf2(e2, out, upper_pred=t3_seg)
        out = out + t2_feat

        # Decoder stage 4 with TSBRF
        out = F.relu(F.interpolate(self.decoder4(out),
                                   scale_factor=2, mode='bilinear', align_corners=False))
        t1_feat, t1_seg, t1_bnd = self.tsbrf1(e1, out, upper_pred=t2_seg)
        out = out + t1_feat

        # Final decoder stage
        out = F.relu(F.interpolate(self.decoder5(out),
                                   scale_factor=2, mode='bilinear', align_corners=False))
        main = F.interpolate(self.final_conv(out), size=input_size,
                             mode='bilinear', align_corners=True)

        # Deep supervision outputs
        if self.gt_ds:
            aux = [
                F.interpolate(self.ds1(t4_feat), size=input_size,
                              mode='bilinear', align_corners=True),
                F.interpolate(self.ds2(t3_feat), size=input_size,
                              mode='bilinear', align_corners=True),
                F.interpolate(self.ds3(t2_feat), size=input_size,
                              mode='bilinear', align_corners=True),
                F.interpolate(self.ds4(t1_feat), size=input_size,
                              mode='bilinear', align_corners=True),
            ]
            boundary_maps = {
                'stage4': F.interpolate(t4_bnd, size=input_size,
                                        mode='bilinear', align_corners=True),
                'stage3': F.interpolate(t3_bnd, size=input_size,
                                        mode='bilinear', align_corners=True),
                'stage2': F.interpolate(t2_bnd, size=input_size,
                                        mode='bilinear', align_corners=True),
                'stage1': F.interpolate(t1_bnd, size=input_size,
                                        mode='bilinear', align_corners=True),
            }
            return aux, main, boundary_maps

        boundary_maps = {
            'stage4': F.interpolate(t4_bnd, size=input_size,
                                    mode='bilinear', align_corners=True),
            'stage3': F.interpolate(t3_bnd, size=input_size,
                                    mode='bilinear', align_corners=True),
            'stage2': F.interpolate(t2_bnd, size=input_size,
                                    mode='bilinear', align_corners=True),
            'stage1': F.interpolate(t1_bnd, size=input_size,
                                    mode='bilinear', align_corners=True),
        }
        return main, boundary_maps

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """KAN regularization loss for sparsity and entropy."""
        total = 0.0
        for m in self.modules():
            if isinstance(m, KANLinear):
                total += m.regularization_loss(regularize_activation, regularize_entropy)
        return total


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*80)
    print("W-KANet: Wavelet-Guided KAN with Tri-Stream Boundary-Reverse Fusion")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Initialize model
    model = WKANet(
        num_classes=1, 
        input_channels=1,
        img_size=256, 
        embed_dims=(320, 512),
        gt_ds=True, 
        drop_path_rate=0.1, 
        depths=(1, 1, 1)
    ).to(device)

    # Test forward pass
    x = torch.randn(2, 1, 256, 256, device=device)
    model.eval()
    with torch.no_grad():
        aux, main, bm = model(x)

    print(f"\n✓ Main output     : {tuple(main.shape)}")
    print(f"✓ Aux outputs     : {len(aux)} predictions")
    print(f"✓ Boundary maps   : {len(bm)} stages")

    # Parameter count
    n = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Total parameters: {n:,} ({n/1e6:.1f}M)")
    
    print("\n" + "="*80)
    print("✅ W-KANet model loaded successfully!")
    print("="*80)
