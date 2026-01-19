import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torch.nn.functional import interpolate
import torchvision.transforms as transforms
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from torchmetrics.image import ErrorRelativeGlobalDimensionlessSynthesis
ergas = ErrorRelativeGlobalDimensionlessSynthesis()  

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, content_feat, style_feat):
        size = content_feat.size()
        content_mean, content_std = (
            content_feat.view(size[0], size[1], -1).mean(2).view(size[0], size[1], 1, 1),
            content_feat.view(size[0], size[1], -1).std(2).view(size[0], size[1], 1, 1) + self.eps,
        )
        style_mean, style_std = (
            style_feat.view(size[0], size[1], -1).mean(2).view(size[0], size[1], 1, 1),
            style_feat.view(size[0], size[1], -1).std(2).view(size[0], size[1], 1, 1) + self.eps,
        )
        normalized = (content_feat - content_mean) / content_std
        return normalized * style_std + style_mean

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 5, 7), "Kernel size must be 3, 5, or 7"
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_out)
        x_out = self.sigmoid(x_out)
        return x * x_out
    
# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self, channel=32):
        super(Resblock, self).__init__()
        
        # Use the channel parameter instead of hardcoding 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, 
                               stride=1, padding=1, bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, 
                               stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv20(x))
        rs1 = self.conv21(rs1)
        rs = torch.add(x, rs1)
        return rs


# --- small helper: convex pan mixer (learnable, but constrained) ---
class ConvexPanMixer(nn.Module):
    """
    Produces a PAN-like channel as a convex combination of RGB bands.
    We parametrize weights with logits -> softmax so they are >=0 and sum to 1.
    init_mode: 'avg' -> [1/3,1/3,1/3], 'luma' -> [0.2989, 0.5870, 0.1140]
    """
    def __init__(self, in_nc=3, init_mode='avg'):
        super().__init__()
        assert in_nc >= 1
        self.in_nc = in_nc
        self.logits = nn.Parameter(torch.zeros(in_nc))
        with torch.no_grad():
            if init_mode == 'luma' and in_nc >= 3:
                w = torch.tensor([0.2989, 0.5870, 0.1140])
                if in_nc > 3:  # distribute the rest equally if more channels
                    extra = torch.full((in_nc-3,), 1e-6)
                    w = torch.cat([w, extra], dim=0)
                w = w / w.sum()
            else:
                w = torch.full((in_nc,), 1.0 / in_nc)
            # inverse softmax init
            self.logits.copy_(torch.log(w))

    def forward(self, x):
        # x: [B, C, H, W]
        if self.in_nc == 1:
            return x
        w = torch.softmax(self.logits, dim=0)  # [C]
        # weighted sum across channel dimension
        pan = torch.sum(x * w.view(1, -1, 1, 1), dim=1, keepdim=True)
        return pan


class MTFNetFusion(nn.Module):
    """
    GLP-FS-in-the-loop + residual learning.
    - pan_lp_i is generated per band with reflect padding, MTF blur, ↓, ↑
    - detail_i = pan - pan_lp_i  (classical GLP-FS detail)
    - detail_hat_i = detail_i + R(detail_i, ms_up_i)  (learn residual detail; last layer linear)
    - gain_i = 1 + Δg(pan, pan_lp_i, mean(ms_up))     (unbounded residual gain)
    - fused = ms_up + gain_i * detail_hat_i, then a light refinement residual

    Args:
        rgb_in_nc: channels of guidance image (3 for S2 RGB)
        ms_in_nc:  number of MS bands in this branch (e.g., 4 for 10m, 6 for 20m)
        out_nc:    equals ms_in_nc (kept for compatibility)
        nf, nb, gc: RRDB/Resblock config
        sensor:    affects default MTF values (kept for init only)
        mtf_ratio: int or list[int] per band; the GLP scale used to form pan_lp
                   (e.g., 4 for Wald 10m branch; 8 for a 20m->160m synthetic pair; 6 for 60m->360m)
        pan_init:  'avg' or 'luma' init for convex mixer
    """
    def __init__(self, rgb_in_nc, ms_in_nc, out_nc, nf=32, nb=4, gc=32,
                 sensor='S2', mtf_ratio=4, pan_init='avg'):
        super().__init__()
        self.ms_in_nc = ms_in_nc
        self.rgb_in_nc = rgb_in_nc
        self.sensor = sensor

        # ---- PAN synthesis: convex, learnable but physically sane ----
        self.pan_mixer = ConvexPanMixer(in_nc=rgb_in_nc, init_mode=pan_init)

        # ---- Per-band MTF filters (kernel size from sigma; reflection pad per band) ----
        # Build once; we still allow learning but they start as proper Gaussian-MTFs.
        mtf_vals = self._default_mtf_values(sensor, ms_in_nc)
        ratios = self._broadcast_ratio(mtf_ratio, ms_in_nc)  # list[int] per band

        self.mtf_filters = nn.ModuleList()
        self.mtf_pads = nn.ModuleList()
        for i in range(ms_in_nc):
            sigma, ksz = self._sigma_and_ksize_from_mtf(mtf_vals[i], ratios[i])
            pad = ksz // 2
            conv = nn.Conv2d(1, 1, kernel_size=ksz, padding=0, bias=False)
            with torch.no_grad():
                kernel = self._gaussian_kernel(ksz, sigma)
                conv.weight.copy_(kernel)
            self.mtf_filters.append(conv)
            self.mtf_pads.append(nn.ReflectionPad2d(pad))
        self.register_buffer('mtf_ratios', torch.tensor(ratios, dtype=torch.int32))

        # ---- Detail residual network (last layer linear!) ----
        self.detail_net = nn.Sequential(
            nn.Conv2d(2, nf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *[RRDB(nf=nf, gc=gc) for _ in range(nb//2)],
            nn.Conv2d(nf, 1, kernel_size=3, padding=1)  # linear output
        )

        # ---- Residual gain estimator (per-band; shared weights, 1-channel output) ----
        # Input: [pan, pan_lp_i, ms_mean] -> Δg_i
        self.gain_estimator = nn.Sequential(
            nn.Conv2d(3, nf // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf // 2, 1, kernel_size=1, padding=0)  # per-band scalar map
        )

        # ---- Light refinement on top of injected result ----
        self.refinement = nn.Sequential(
            nn.Conv2d(ms_in_nc + 1, nf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *[Resblock(channel=nf) for _ in range(nb//2)],
            nn.Conv2d(nf, ms_in_nc, kernel_size=3, padding=1)
        )

    # ---------- init helpers ----------
    def _default_mtf_values(self, sensor, C):
        if sensor == 'WV2':
            v = [0.32, 0.26, 0.28, 0.24, 0.22, 0.23, 0.23, 0.20]
        elif sensor == 'IKONOS':
            v = [0.26, 0.28, 0.29, 0.28]
        else:  # Sentinel-2 (approx)
            v = [0.29, 0.30, 0.28, 0.24, 0.23, 0.23, 0.25, 0.25, 0.27, 0.28, 0.30, 0.30]
        if len(v) < C:
            v = v + [0.28] * (C - len(v))
        return v[:C]

    def _broadcast_ratio(self, ratio, C):
        if isinstance(ratio, int):
            return [ratio] * C
        assert isinstance(ratio, (list, tuple)) and len(ratio) == C, \
            "mtf_ratio must be int or list/tuple of length ms_in_nc"
        return list(map(int, ratio))

    def _sigma_and_ksize_from_mtf(self, mtf_value, ratio):
        # sigma derived from MTF at Nyquist (same as your classical code)
        sigma = float(-ratio * np.sqrt(2) * np.log(mtf_value) / (2 * np.pi ** 2))
        # robust kernel size: cover ~4 sigmas; ensure odd and >=3
        ksz = max(3, 2 * int(4 * sigma + 0.5) + 1)
        return sigma, ksz

    def _gaussian_kernel(self, ksz, sigma):
        ax = torch.arange(ksz) - (ksz // 2)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        ker = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        ker = ker / ker.sum()
        return ker.view(1, 1, ksz, ksz)

    # ---------- GLP-FS low-pass per band ----------
    # ---------- GLP-FS low-pass per band (robust sizing) ----------
    def _simulate_lp(self, pan, filt, pad_layer, ratio: int):
        """
        reflect-pad -> blur (size preserved) -> ↓ to integer LR size -> ↑ back to (H,W)
        Using explicit 'size' avoids PyTorch scale_factor rounding drift (510 vs 512).
        """
        # blur at full res (size preserved because of reflect pad)
        pan_b = filt(pad_layer(pan))                     # [B,1,H,W]
        if ratio > 1:
            _, _, H, W = pan_b.shape
            Hlr = max(1, H // ratio)
            Wlr = max(1, W // ratio)
            # ↓ area to exact LR lattice
            pan_b = F.interpolate(pan_b, size=(Hlr, Wlr), mode='area')
            # ↑ bicubic exactly back to (H,W)
            pan_b = F.interpolate(pan_b, size=(H, W), mode='bicubic', align_corners=False)
        return pan_b

    def forward(self, rgb, ms):
        # --- PAN synthesis (convex convex-combo) ---
        pan = self.pan_mixer(rgb)  # [B,1,H,W]

        # --- Upsample MS if needed: check both H and W ---
        if ms.shape[-2:] != pan.shape[-2:]:
            ms_up = F.interpolate(ms, size=pan.shape[-2:], mode='bilinear', align_corners=False)
        else:
            ms_up = ms

        B, C, H, W = ms_up.shape
        ms_mean = ms_up.mean(dim=1, keepdim=True)

        fused_bands, detail_hats = [], []

        for i in range(self.ms_in_nc):
            ratio_i = int(self.mtf_ratios[i].item())
            pan_lp_i = self._simulate_lp(pan, self.mtf_filters[i], self.mtf_pads[i], ratio_i)
            # shapes match pan exactly -> no broadcast error
            detail_i = pan - pan_lp_i

            # learn residual detail (linear head)
            res_d_i = self.detail_net(torch.cat([detail_i, ms_up[:, i:i+1]], dim=1))
            detail_hat_i = detail_i + res_d_i
            
            # residual gain around 1
            dgi = self.gain_estimator(torch.cat([pan, pan_lp_i, ms_mean], dim=1))
            gain_i = 1.0 + dgi

            band_i = ms_up[:, i:i+1] + gain_i * detail_hat_i
            fused_bands.append(band_i)
            detail_hats.append(detail_hat_i)

        enhanced_ms = torch.cat(fused_bands, dim=1)
        mean_detail = torch.mean(torch.cat(detail_hats, dim=1), dim=1, keepdim=True)
        refined = self.refinement(torch.cat([enhanced_ms, mean_detail], dim=1))
        fused = enhanced_ms + refined
        return fused
class FusionRRDBNet(nn.Module):
    def __init__(self, rgb_in_nc, ms_in_nc, out_nc, nf=64, nb=3, gc=32, use_adain=False, use_channel_attention=False, use_spatial_attention=False):
        super(FusionRRDBNet, self).__init__()
        RRDB_block_f = lambda: RRDB(nf=nf, gc=gc)
        self.use_adain = use_adain
        #if use_adain:
        self.adain = AdaIN()
        # use drop out
        self.rgb_dropout = nn.Dropout(p=0.2)
        # channel attention
        self.ms_channel_attention = ChannelAttention(num_channels=nf)
        self.rgb_channel_attention = ChannelAttention(num_channels=nf)
        # spatial attention
        self.spatial_attention_rgb = SpatialAttention(kernel_size=7)
        self.spatial_attention_ms = SpatialAttention(kernel_size=7)

        # RGB encoder branch
        self.rgb_conv_first = nn.Conv2d(rgb_in_nc, nf, 3, 1, 1)
        # self.rgb_down1 = nn.Conv2d(nf, nf, 3, stride=2, padding=1)
        # self.rgb_down2 = nn.Conv2d(nf, nf, 3, stride=2, padding=1)
        self.rgb_rrdb = nn.Sequential(*[RRDB_block_f() for _ in range(nb)])

        # MS branch
        self.ms_conv_first = nn.Conv2d(ms_in_nc, nf, 3, 1, 1)
        # self.ms_down1 = nn.Conv2d(nf, nf, 3, stride=2, padding=1)
        # self.ms_down2 = nn.Conv2d(nf, nf, 3, stride=2, padding=1)
        self.ms_rrdb = nn.Sequential(*[RRDB_block_f() for _ in range(nb)])

        # Fusion
        self.fusion_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.fusion_rrdb = nn.Sequential(*[RRDB_block_f() for _ in range(nb)])

        # Upsample and final output
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, rgb, ms):
        rgb_feat = self.lrelu(self.rgb_conv_first(rgb))
        rgb_feat = self.rgb_rrdb(rgb_feat)
        rgb_feat = self.rgb_dropout(rgb_feat)  # Apply dropout to RGB features
        rgb_feat = self.rgb_channel_attention(rgb_feat)
        rgb_feat = self.spatial_attention_rgb(rgb_feat)

        ms_feat = self.lrelu(self.ms_conv_first(ms))
        ms_feat = self.ms_rrdb(ms_feat)
        ms_feat = self.ms_channel_attention(ms_feat) # helps
        ms_feat = self.spatial_attention_ms(ms_feat) # helps

        # if self.use_adain:

        fused_feat = torch.cat((ms_feat, rgb_feat), dim=1)
        fused_feat = self.lrelu(self.fusion_conv(fused_feat))

        fused_feat = self.fusion_rrdb(fused_feat)
        #fused_feat = self.adain(fused_feat, ms_feat)  # helps not 
        # Transfer MS style to fused features

        fea = self.lrelu(self.upconv1(F.interpolate(fused_feat, scale_factor=1, mode="nearest")))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=1, mode="nearest")))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


def gaussian_blur(x, sigma, kernel_size=5):
    """Apply Gaussian blur to a tensor"""
    blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=sigma)
    return blur(x)

def boxcar_downsample(x, scale_factor):
    """Downsample a tensor using boxcar averaging"""
    kernel_size = int(1 / scale_factor)
    avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size, padding=0)
    return avg_pool(x)

def interpolate(x, scale_factor, mode='bilinear', blur=False, sigma=None, boxcar=False, gaussian_kernel_size=5):
    """
    Downscales a tensor with options for Gaussian blur, boxcar averaging, and interpolation.

    Args:
        x (torch.Tensor): Input tensor.
        scale_factor (float): Downscaling factor (e.g., 1/2 for half the size).
        mode (str): Interpolation mode ('bilinear', 'nearest', etc.).
        blur (bool): Whether to apply Gaussian blur before downsampling.
        sigma (float, optional): Standard deviation of the Gaussian kernel. If None, it's automatically set to 1/scale_factor if blur is True.
        boxcar (bool): Whether to apply boxcar averaging after blurring (if blur is True).
        gaussian_kernel_size (int): Kernel size for gaussian blur
    Returns:
        torch.Tensor: Downscaled tensor.
    """
    if blur:
        if sigma is None:
            sigma = 1 / scale_factor  # Set sigma based on scale factor if not provided
        x = gaussian_blur(x, sigma, gaussian_kernel_size)  # Apply Gaussian blur
    if boxcar:
        x = boxcar_downsample(x, scale_factor)  # Apply boxcar averaging
    else:
        x = nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)  # Apply interpolation
    return x


def make_norm_grid(tensor):
    """Helper function to create normalized visualization grid"""
    if tensor.shape[1] == 2:
        # For 60m: Duplicate first channel for R&G, use second for B
        grid = torchvision.utils.make_grid(
            torch.cat([
                torch.unsqueeze(tensor[:6, 1, :, :], 1),  # Red
                torch.unsqueeze(tensor[:6, 1, :, :], 1),  # Green 
                torch.unsqueeze(tensor[:6, 0, :, :], 1)   # Blue
            ], dim=1)
        )
    elif tensor.shape[1] >= 3:
        grid = torchvision.utils.make_grid(
            torch.cat([
                torch.unsqueeze(tensor[:6, 2, :, :], 1),
                torch.unsqueeze(tensor[:6, 1, :, :], 1),
                torch.unsqueeze(tensor[:6, 0, :, :], 1)
            ], dim=1)
        )
    else:
        grid = torchvision.utils.make_grid(tensor[:6])
    return (grid - grid.min()) / (grid.max() - grid.min())


def gram_schmidt_fusion(ms_img, pan_img):
    """
    Apply Gram-Schmidt pan-sharpening
    Args:
        ms_img: Multispectral image tensor (B, C, H, W)
        pan_img: RGB or Pan image tensor (B, 3/1, H, W)
    Returns:
        Pansharpened image tensor (B, C, H, W)
    """
    # Convert RGB pan to grayscale if needed
    if pan_img.shape[1] == 3:
        pan_img = 0.2989 * pan_img[:,0:1] + 0.5870 * pan_img[:,1:2] + 0.1140 * pan_img[:,2:3]
    
    # Ensure spatial dimensions match
    if pan_img.shape[-2:] != ms_img.shape[-2:]:
        pan_img = nn.functional.interpolate(pan_img, size=ms_img.shape[-2:], mode='bilinear', align_corners=False)
    
    # Calculate mean of each band
    ms_mean = ms_img.mean(dim=(2, 3), keepdim=True)
    
    # Center the data
    ms_centered = ms_img - ms_mean
    
    # Create synthetic pan by averaging MS bands
    synthetic_pan = ms_img.mean(dim=1, keepdim=True)
    
    # Prepare tensors for covariance calculation
    ms_centered_flat = ms_centered.flatten(start_dim=2)  # (B, C, H*W)
    synthetic_pan_flat = synthetic_pan.flatten(start_dim=2)  # (B, 1, H*W)
    
    # Calculate covariance and variance
    covariance = torch.bmm(ms_centered_flat, synthetic_pan_flat.transpose(1, 2))  # (B, C, 1)
    covariance = covariance / ms_centered_flat.shape[2]  # Normalize by number of pixels
    
    variance = synthetic_pan_flat.var(dim=2, keepdim=True, unbiased=False) + 1e-6
    
    # Calculate GS coefficients
    coefficients = (covariance / variance).unsqueeze(-1)  # (B, C, 1, 1)
    
    # Apply GS transformation
    pansharpened = ms_centered + coefficients * (pan_img - synthetic_pan)
    pansharpened = pansharpened + ms_mean
    
    return pansharpened

class FusionNetwork(pl.LightningModule):
    def __init__(self,mode='train',GSD='GSD'):
        super(FusionNetwork, self).__init__()
        # self.rrdb_block = RRDB(channels=64)  # Example channel number, adjust as necessary
        self.hparams.lr = 1e-4
        

        
        self.rrdb_block_10_fusion = MTFNetFusion(3, 4, 4, nf=32, nb=3, gc=16, sensor='S2', mtf_ratio=4, pan_init='luma')
        self.rrdb_block_20_fusion = MTFNetFusion(3, 6, 6, nf=32, nb=3, gc=16, sensor='S2', mtf_ratio=8, pan_init='luma')
        self.rrdb_block_60_fusion = MTFNetFusion(3, 2, 2, nf=32, nb=3, gc=16, sensor='S2', mtf_ratio=24, pan_init='luma')



        self.mode = mode
        #self.GSD = GSD
        self.l1_criterion = nn.L1Loss()
        # self.l1_ergas_criterion = L1ErgasLoss(ratio=4)  # Example ratio, adjust as necessary
        # self.l1_ergas_criterion_20 = L1ErgasLoss(ratio=8)  # Example ratio, adjust as necessary
        # self.l1_ergas_criterion_60 = L1ErgasLoss(ratio=24)  # Example ratio, adjust as necessary

    def forward(self, inputs, fusion_signal=None):
        # Extract input bands
        try:
            x1 = inputs['S2:Red']
            x2 = inputs['S2:Green']
            x3 = inputs['S2:Blue']
            x4 = inputs['S2:NIR']
            x5 = inputs['S2:RE1']
            x6 = inputs['S2:RE2']
            x7 = inputs['S2:RE3']
            x8 = inputs['S2:RE4']
            x9 = inputs['S2:SWIR1']
            x10 = inputs['S2:SWIR2']
            x11 = inputs['S2:CoastAerosal']
            x12 = inputs['S2:WaterVapor']
        
        except:
            x1 = inputs[:, 3, :, :].unsqueeze(1)  # S2:Red (Band 4)
            x2 = inputs[:, 2, :, :].unsqueeze(1)  # S2:Green (Band 3)
            x3 = inputs[:, 1, :, :].unsqueeze(1)  # S2:Blue (Band 2)
            x4 = inputs[:, 7, :, :].unsqueeze(1)  # S2:NIR (Band 8)
            x5 = inputs[:, 4, :, :].unsqueeze(1)  # S2:RE1 (Band 5)
            x6 = inputs[:, 5, :, :].unsqueeze(1)  # S2:RE2 (Band 6)
            x7 = inputs[:, 6, :, :].unsqueeze(1)  # S2:RE3 (Band 7)
            x8 = inputs[:, 8, :, :].unsqueeze(1)  # S2:RE4 (Band 8A) # 
            x9 = inputs[:, 10, :, :].unsqueeze(1) # S2:SWIR1 (Band 11)
            x10 = inputs[:, 11, :, :].unsqueeze(1) # S2:SWIR2 (Band 12)
            x11 = inputs[:, 0, :, :].unsqueeze(1) # S2:CoastAerosal (Band 1)
            x12 = inputs[:, 9, :, :].unsqueeze(1) # S2:WaterVapor (Band 9)


        #print dict inputs
        #print(inputs.keys())
        # Training mode
        if self.mode == 'train':
            # 10m branch
            fusion_signal_10 = torch.cat([x1, x2, x3], dim=1)
            inputs_ms_10m_40m = interpolate(torch.cat([x1, x2, x3, x4], dim=1), 1/4, blur=True, boxcar=True)
            x_40m_to_10m = interpolate(inputs_ms_10m_40m, 4)
            del inputs_ms_10m_40m
            
            # 20m branch
            fusion_signal_20 = interpolate(torch.cat([x1, x2, x3], dim=1), 1/2, blur=True, boxcar=True)
            inputs_ms_20m_160m = interpolate(torch.cat([x5, x6, x7, x8, x9, x10], dim=1), 1/8, blur=True, boxcar=True)
            x_160m_to_20m = interpolate(inputs_ms_20m_160m, 8)
            del inputs_ms_20m_160m

            # 60m branch
            fusion_signal_60 = interpolate(torch.cat([x1, x2, x3], dim=1), 1/6, blur=True, boxcar=True)
            inputs_ms_60m_1440m = interpolate(torch.cat([x11, x12], dim=1), 1/24, blur=True, boxcar=True)
            x_1440m_to_60m = interpolate(inputs_ms_60m_1440m, 24)
            del inputs_ms_60m_1440m

        # Evaluation mode
        else:
            # Use provided fusion signal or create one
            if fusion_signal is None:
                fusion_signal = interpolate(torch.cat([x1, x2, x3], dim=1), 4)
            else:
                # Create a new tensor instead of reassigning
                try:
                    new_fusion_signal = torch.cat([fusion_signal['R'], fusion_signal['G'], fusion_signal['B']], dim=1)
                except:
                    new_fusion_signal = fusion_signal

                del fusion_signal  # Delete the old fusion_signal
                fusion_signal = new_fusion_signal
                del new_fusion_signal

            # Use the same fusion signal for all branches
            fusion_signal_60 = fusion_signal_20 = fusion_signal_10 = fusion_signal



                                          
            # Interpolate input bands (eval mode: upsample by native ratio)
            x_40m_to_10m = interpolate(torch.cat([x1, x2, x3, x4], dim=1), 4)
            x_160m_to_20m = interpolate(torch.cat([x5, x6, x7, x8, x9, x10], dim=1), 8)
            x_1440m_to_60m = interpolate(torch.cat([x11, x12], dim=1), 24)

        # 10m branch
        x_ms_10 = x_40m_to_10m
        output_10 = self.rrdb_block_10_fusion(fusion_signal_10, x_ms_10)
        GT_out_10 = torch.cat([x1, x2, x3, x4], dim=1)
        del x_ms_10, fusion_signal_10, x_40m_to_10m

        # 20m branch output
        x_ms_20 = x_160m_to_20m
        output_20 = self.rrdb_block_20_fusion(fusion_signal_20, x_ms_20)
        GT_out_20 = torch.cat([x5, x6, x7, x8, x9, x10], dim=1)
        del x_ms_20, fusion_signal_20, x_160m_to_20m

        # 60m branch output
        x_ms_60 = x_1440m_to_60m
        output_60 = self.rrdb_block_60_fusion(fusion_signal_60, x_ms_60)
        GT_out_60 = torch.cat([x11, x12], dim=1)
        del x_ms_60, fusion_signal_60, x_1440m_to_60m


        return output_10, output_20, output_60, GT_out_10, GT_out_20, GT_out_60

    def training_step(self, batch, batch_idx):
        # Forward pass returns outputs and ground truth for all resolutions

        output_10, output_20, output_60, GT_out_10, GT_out_20, GT_out_60 = self(batch, fusion_signal=None)

        # Calculate combined L1 loss across all resolutions
        loss = (self.l1_criterion(output_10, GT_out_10) + 
                self.l1_criterion(output_20, GT_out_20) + 
                self.l1_criterion(output_60, GT_out_60))
        
        # Calculate L1 ERGAS loss for each resolution

        self.log("loss_train", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        if batch_idx == 0:
            self.trainer.train_dataloader.dataset.epoch += 1
        return loss
    
    def on_validation_start(self):
        """Called when the validation loop begins."""
        self.mode = 'eval'
        self.eval()  # Set model to evaluation mode

    def on_validation_end(self):
        """Called when the validation loop ends."""
        self.mode = 'train'
        self.train()  # Set model back to training mode

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            # Forward pass (keep this to test if the problem is in the forward pass)
            output_10, output_20, output_60, GT_out_10, GT_out_20, GT_out_60 = self(batch, fusion_signal=None)

            # Dummy metric to satisfy Lightning's requirement
            dummy_metric = torch.tensor(9.0, device=self.device)  # Ensure it's on the correct device
            #self.log('metric_val', dummy_metric, prog_bar=False, sync_dist=True)

            # Interpolate ground truth outputs
            GT_out_10_p = interpolate(GT_out_10, 4)
            GT_out_20_p = interpolate(GT_out_20, 8)
            GT_out_60_p = interpolate(GT_out_60, 24)

            # Calculate GS outputs
            x_rgb = torch.cat([batch['S2:Red'], batch['S2:Green'], batch['S2:Blue']], dim=1)
            pan_2_5m = interpolate(x_rgb, 4)  # Simulate 2.5m pan image
            del x_rgb  # Free memory

            gs_output_10 = gram_schmidt_fusion(GT_out_10_p, pan_2_5m)
            gs_output_20 = gram_schmidt_fusion(GT_out_20_p, pan_2_5m)
            gs_output_60 = gram_schmidt_fusion(GT_out_60_p, pan_2_5m)
            del pan_2_5m  # Free memory



            # ergas = ErrorRelativeGlobalDimensionlessSynthesis(reduction=None).to('cpu')
            # #Calculate metrics
            metric_10 = ergas(interpolate(output_10,1/4, blur=True, boxcar=True).detach().cpu(), GT_out_10.detach().cpu()).round()
            metric_20 = ergas(interpolate(output_20,1/8, blur=True, boxcar=True).detach().cpu(), GT_out_20.detach().cpu()).round()
            metric_60 = ergas(interpolate(output_60,1/24, blur=True, boxcar=True).detach().cpu(), GT_out_60.detach().cpu()).round()
            combined_metric = (metric_10 + metric_20 + metric_60) / 3

            gs_metric_10 = ergas(interpolate(gs_output_10,1/4, blur=True, boxcar=True).detach().cpu(), GT_out_10.detach().cpu()).round()
            gs_metric_20 = ergas(interpolate(gs_output_20,1/8, blur=True, boxcar=True).detach().cpu(), GT_out_20.detach().cpu()).round()
            gs_metric_60 = ergas(interpolate(gs_output_60,1/24, blur=True, boxcar=True).detach().cpu(), GT_out_60.detach().cpu()).round()
            gs_combined_metric = (gs_metric_10 + gs_metric_20 + gs_metric_60) / 3

            # GS is baseline so we subtract from metric 
            subtracted_metric_10 = metric_10 - gs_metric_10
            subtracted_metric_20 = metric_20 - gs_metric_20
            subtracted_metric_60 = metric_60 - gs_metric_60
            # Log all metrics
            metrics = {
                "metric_val_10m": metric_10.item(),
                "metric_val_20m": metric_20.item(),
                "metric_val_60m": metric_60.item(),
                "metric_val": combined_metric.item(),
                "gs_metric_val_10m": gs_metric_10.item(),
                "gs_metric_val_20m": gs_metric_20.item(),
                "gs_metric_val_60m": gs_metric_60.item(),
                "gs_metric_val": gs_combined_metric.item(),
                "subtracted_metric_10m": subtracted_metric_10.item(),
                "subtracted_metric_20m": subtracted_metric_20.item(),
                "subtracted_metric_60m": subtracted_metric_60.item(),
            }

            # Log metrics in batch
            for name, value in metrics.items():
                self.log(name, value, prog_bar=name=="metric_val", sync_dist=True, on_epoch=True)

            # Only process images for first batch
            #print('batch_idx',batch_idx)
            if batch_idx == 0:
                # Extract first 6 samples of each band
                bands = {
                    'Red': batch['S2:Red'][:6],
                    'Green': batch['S2:Green'][:6],
                    'Blue': batch['S2:Blue'][:6],
                    'RE1': batch['S2:RE1'][:6],
                    'RE2': batch['S2:RE2'][:6],
                    'RE3': batch['S2:RE3'][:6],
                    'Coast': batch['S2:CoastAerosal'][:6],
                    'Water': batch['S2:WaterVapor'][:6]
                }

                # Create and normalize visualization grids
                grids = {
                    'rgb': torchvision.utils.make_grid(torch.cat([bands['Blue'], bands['Green'], bands['Red']], dim=1).cpu()),
                    're': torchvision.utils.make_grid(torch.cat([bands['RE3'], bands['RE2'], bands['RE1']], dim=1).cpu()),
                    'coast': torchvision.utils.make_grid(torch.cat([bands['Water'], bands['Water'], bands['Coast']], dim=1).cpu())
                }
                del bands  # Free memory

                # Normalize and log input images
                for name, grid in grids.items():
                    norm_grid = (grid - grid.min()) / (grid.max() - grid.min())
                    self.logger.experiment.add_image(f'Input/{name.upper()}', norm_grid, self.current_epoch)
                    del grid, norm_grid  # Free memory
                del grids

                # Log output images
                outputs = {
                    '10m': (output_10.cpu(), GT_out_10_p.cpu(), gs_output_10.cpu()),
                    '20m': (output_20.cpu(), GT_out_20_p.cpu(), gs_output_20.cpu()),
                    '60m': (output_60.cpu(), GT_out_60_p.cpu(), gs_output_60.cpu())
                }

                for res, (pred, gt, gs) in outputs.items():
                    self.logger.experiment.add_image(f'Output/{res}/Pred', make_norm_grid(pred).cpu(), self.current_epoch)
                    self.logger.experiment.add_image(f'Output/{res}/GT', make_norm_grid(gt).cpu(), self.current_epoch)
                    self.logger.experiment.add_image(f'GS/{res}/Output', make_norm_grid(gs).cpu(), self.current_epoch)
                    del pred, gt, gs
                del outputs


                #pass

            # Cleanup remaining tensors
            del metric_10, metric_20, metric_60, combined_metric, gs_metric_10, gs_metric_20, gs_metric_60, gs_combined_metric

            del output_10, output_20, output_60
            del GT_out_10, GT_out_20, GT_out_60
            del gs_output_10, gs_output_20, gs_output_60

            return dummy_metric  # Return a dummy metric
    

    
    def predict_step(self, batch, batch_idx, fusion_signal=None):
        # Get outputs from all three branches
        output_10, output_20, output_60, GT_out_10, GT_out_20, GT_out_60 = self(batch, fusion_signal=fusion_signal)
        
        # Helper function to create normalized grid (reusing from training)


        # Log all three resolutions
        self.logger.experiment.add_image('Predict/10m/Output', make_norm_grid(output_10), batch_idx)
        self.logger.experiment.add_image('Predict/10m/GT', make_norm_grid(GT_out_10), batch_idx)
        self.logger.experiment.add_image('Predict/20m/Output', make_norm_grid(output_20), batch_idx)
        self.logger.experiment.add_image('Predict/20m/GT', make_norm_grid(GT_out_20), batch_idx)
        self.logger.experiment.add_image('Predict/60m/Output', make_norm_grid(output_60), batch_idx)
        self.logger.experiment.add_image('Predict/60m/GT', make_norm_grid(GT_out_60), batch_idx)
        # Return all outputs for GSD mode
        return output_10, output_20, output_60, GT_out_10, GT_out_20, GT_out_60

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
