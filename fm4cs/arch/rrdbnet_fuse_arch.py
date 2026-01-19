import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
# ablation: use adain or not
# use drop out or not
#use noise augmention for fusion signal or 
# channel attention
# use spatial attention

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
    
class FusionRRDBNet(nn.Module):
    def __init__(self, rgb_in_nc, ms_in_nc, out_nc, nf=64, nb=3, gc=32, use_adain=False, use_channel_attention=False, use_spatial_attention=False):
        super(FusionRRDBNet, self).__init__()
        RRDB_block_f = lambda: RRDB(nf=nf, gc=gc)
        self.use_adain = use_adain
        #if use_adain:
        self.adain = AdaIN()
        # use drop out
        #self.rgb_dropout = nn.Dropout(p=0.2)
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
        #rgb_feat = self.rgb_dropout(rgb_feat)  # Apply dropout to RGB features
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


class ConvexPanMixer(nn.Module):
    """
    Produces a PAN-like channel as a convex combination of RGB bands.
    We parametrize weights with logits -> softmax so they are >=0 and sum to 1.
    init_mode: 'avg' -> [1/3,1/3,1/3], 'luma' -> [0.2989, 0.5870, 0.1140]
    """

    def __init__(self, in_nc=3, init_mode="avg"):
        super().__init__()
        assert in_nc >= 1
        self.in_nc = in_nc
        self.logits = nn.Parameter(torch.zeros(in_nc))
        with torch.no_grad():
            if init_mode == "luma" and in_nc >= 3:
                w = torch.tensor([0.2989, 0.5870, 0.1140])
                if in_nc > 3:
                    extra = torch.full((in_nc - 3,), 1e-6)
                    w = torch.cat([w, extra], dim=0)
                w = w / w.sum()
            else:
                w = torch.full((in_nc,), 1.0 / in_nc)
            self.logits.copy_(torch.log(w))

    def forward(self, x):
        if self.in_nc == 1:
            return x
        w = torch.softmax(self.logits, dim=0)
        return torch.sum(x * w.view(1, -1, 1, 1), dim=1, keepdim=True)


class Resblock(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        self.conv20 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv21 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv20(x))
        rs1 = self.conv21(rs1)
        return torch.add(x, rs1)


class MTFNetFusion(nn.Module):
    """
    GLP-FS-in-the-loop + residual learning.
    - pan_lp_i is generated per band with reflect padding, MTF blur, ↓, ↑
    - detail_i = pan - pan_lp_i
    - detail_hat_i = detail_i + R(detail_i, ms_up_i)
    - gain_i = 1 + Δg(pan, pan_lp_i, mean(ms_up))
    - fused = ms_up + gain_i * detail_hat_i, then a light refinement residual
    """

    def __init__(
        self,
        rgb_in_nc,
        ms_in_nc,
        out_nc,
        nf=32,
        nb=4,
        gc=32,
        sensor="S2",
        mtf_ratio=4,
        pan_init="avg",
    ):
        super().__init__()
        self.ms_in_nc = ms_in_nc
        self.rgb_in_nc = rgb_in_nc

        self.pan_mixer = ConvexPanMixer(in_nc=rgb_in_nc, init_mode=pan_init)

        mtf_vals = self._default_mtf_values(sensor, ms_in_nc)
        ratios = self._broadcast_ratio(mtf_ratio, ms_in_nc)

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
        self.register_buffer("mtf_ratios", torch.tensor(ratios, dtype=torch.int32))

        self.detail_net = nn.Sequential(
            nn.Conv2d(2, nf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *[RRDB(nf=nf, gc=gc) for _ in range(nb // 2)],
            nn.Conv2d(nf, 1, kernel_size=3, padding=1),
        )

        self.gain_estimator = nn.Sequential(
            nn.Conv2d(3, nf // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf // 2, 1, kernel_size=1, padding=0),
        )

        self.refinement = nn.Sequential(
            nn.Conv2d(ms_in_nc + 1, nf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *[Resblock(channel=nf) for _ in range(nb // 2)],
            nn.Conv2d(nf, ms_in_nc, kernel_size=3, padding=1),
        )

    def _default_mtf_values(self, sensor, C):
        if sensor == "WV2":
            v = [0.32, 0.26, 0.28, 0.24, 0.22, 0.23, 0.23, 0.20]
        elif sensor == "IKONOS":
            v = [0.26, 0.28, 0.29, 0.28]
        else:
            v = [0.29, 0.30, 0.28, 0.24, 0.23, 0.23, 0.25, 0.25, 0.27, 0.28, 0.30, 0.30]
        if len(v) < C:
            v = v + [0.28] * (C - len(v))
        return v[:C]

    def _broadcast_ratio(self, ratio, C):
        if isinstance(ratio, int):
            return [ratio] * C
        assert isinstance(ratio, (list, tuple)) and len(ratio) == C, "mtf_ratio must be int or list/tuple of length ms_in_nc"
        return list(map(int, ratio))

    def _sigma_and_ksize_from_mtf(self, mtf_value, ratio):
        sigma = float(-ratio * np.sqrt(2) * np.log(mtf_value) / (2 * np.pi ** 2))
        ksz = max(3, 2 * int(4 * sigma + 0.5) + 1)
        return sigma, ksz

    def _gaussian_kernel(self, ksz, sigma):
        ax = torch.arange(ksz) - (ksz // 2)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        ker = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        ker = ker / ker.sum()
        return ker.view(1, 1, ksz, ksz)

    def _simulate_lp(self, pan, filt, pad_layer, ratio: int):
        pan_b = filt(pad_layer(pan))
        if ratio > 1:
            _, _, H, W = pan_b.shape
            Hlr = max(1, H // ratio)
            Wlr = max(1, W // ratio)
            pan_b = F.interpolate(pan_b, size=(Hlr, Wlr), mode="area")
            pan_b = F.interpolate(pan_b, size=(H, W), mode="bicubic", align_corners=False)
        return pan_b

    def forward(self, rgb, ms):
        pan = self.pan_mixer(rgb)

        if ms.shape[-2:] != pan.shape[-2:]:
            ms_up = F.interpolate(ms, size=pan.shape[-2:], mode="bilinear", align_corners=False)
        else:
            ms_up = ms

        ms_mean = ms_up.mean(dim=1, keepdim=True)

        fused_bands, detail_hats = [], []
        for i in range(self.ms_in_nc):
            ratio_i = int(self.mtf_ratios[i].item())
            pan_lp_i = self._simulate_lp(pan, self.mtf_filters[i], self.mtf_pads[i], ratio_i)
            detail_i = pan - pan_lp_i

            res_d_i = self.detail_net(torch.cat([detail_i, ms_up[:, i : i + 1]], dim=1))
            detail_hat_i = detail_i + res_d_i

            dgi = self.gain_estimator(torch.cat([pan, pan_lp_i, ms_mean], dim=1))
            gain_i = 1.0 + dgi

            band_i = ms_up[:, i : i + 1] + gain_i * detail_hat_i
            fused_bands.append(band_i)
            detail_hats.append(detail_hat_i)

        enhanced_ms = torch.cat(fused_bands, dim=1)
        mean_detail = torch.mean(torch.cat(detail_hats, dim=1), dim=1, keepdim=True)
        refined = self.refinement(torch.cat([enhanced_ms, mean_detail], dim=1))
        return enhanced_ms + refined
