import torch
import torch.nn as nn
import torch.nn.functional as F


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
