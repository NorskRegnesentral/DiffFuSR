import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torch.nn.functional import interpolate
import torchvision.transforms as transforms
from torch import Tensor
import torch.nn.functional as F


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
        
        self.rrdb_block_60_fusion = FusionRRDBNet(3, 12, 2, nf=64, nb=5, gc=32, use_adain=False)
        self.rrdb_block_20_fusion = FusionRRDBNet(3, 10, 6, nf=64, nb=5, gc=32, use_adain=False)
        self.rrdb_block_10_fusion = FusionRRDBNet(3, 4, 4, nf=64, nb=5, gc=32, use_adain=False)

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


        # Training mode
        if self.mode == 'train':
            # 10m branch
            fusion_signal_10 = torch.cat([x1, x2, x3], dim=1)
            inputs_ms_10m_40m = interpolate(torch.cat([x1, x2, x3, x4], dim=1), 1/4, blur=True, boxcar=True)
            x_40m_to_10m = interpolate(inputs_ms_10m_40m, 4)
            del inputs_ms_10m_40m
            
            # 20m branch
            fusion_signal_20 = interpolate(torch.cat([x1, x2, x3], dim=1), 1/2, blur=True, boxcar=True)
            inputs_ms_10m_80m = interpolate(torch.cat([x1, x2, x3, x4], dim=1), 1/8, blur=True, boxcar=True)
            inputs_ms_20m_160m = interpolate(torch.cat([x5, x6, x7, x8, x9, x10], dim=1), 1/8, blur=True, boxcar=True)
            x_80m_to_20m = interpolate(inputs_ms_10m_80m, 4)
            del inputs_ms_10m_80m
            x_160m_to_20m = interpolate(inputs_ms_20m_160m, 8)
            del inputs_ms_20m_160m

            # 60m branch
            fusion_signal_60 = interpolate(torch.cat([x1, x2, x3], dim=1), 1/6, blur=True, boxcar=True)
            inputs_ms_10m_240m = interpolate(torch.cat([x1, x2, x3, x4], dim=1), 1/24, blur=True, boxcar=True)
            inputs_ms_20m_480m = interpolate(torch.cat([x5, x6, x7, x8, x9, x10], dim=1), 1/24, blur=True, boxcar=True)
            inputs_ms_60m_1440m = interpolate(torch.cat([x11, x12], dim=1), 1/24, blur=True, boxcar=True)
            x_240m_to_60m = interpolate(inputs_ms_10m_240m, 4)
            del inputs_ms_10m_240m
            x_480m_to_60m = interpolate(inputs_ms_20m_480m, 8)
            del inputs_ms_20m_480m
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

            # Interpolate input bands
            x_40m_to_10m = interpolate(torch.cat([x1, x2, x3, x4], dim=1), 4) #interpolate(torch.cat([x1, x2, x3, x4], dim=1), 4)
            x_80m_to_20m = interpolate(torch.cat([x1, x2, x3, x4], dim=1), 4)#interpolate(torch.cat([x1, x2, x3, x4], dim=1), 4)
            x_160m_to_20m = interpolate(torch.cat([x5, x6, x7, x8, x9, x10], dim=1), 4)#interpolate(torch.cat([x5, x6, x7, x8, x9, x10], dim=1), 8)
            x_240m_to_60m = interpolate(torch.cat([x1, x2, x3, x4], dim=1), 4)#interpolate(torch.cat([x1, x2, x3, x4], dim=1), 4)
            x_480m_to_60m = interpolate(torch.cat([x5, x6, x7, x8, x9, x10], dim=1), 4) #interpolate(torch.cat([x5, x6, x7, x8, x9, x10], dim=1), 8)
            x_1440m_to_60m = interpolate(torch.cat([x11, x12], dim=1), 4) #interpolate(torch.cat([x11, x12], dim=1), 24)

        # 10m branch
        x_ms_10 = x_40m_to_10m
        output_10 = self.rrdb_block_10_fusion(fusion_signal_10, x_ms_10)
        GT_out_10 = torch.cat([x1, x2, x3, x4], dim=1)
        del x_ms_10, fusion_signal_10, x_40m_to_10m

        # 20m branch output
        x_ms_20 = torch.cat([x_80m_to_20m, x_160m_to_20m], dim=1)
        output_20 = self.rrdb_block_20_fusion(fusion_signal_20, x_ms_20)
        GT_out_20 = torch.cat([x5, x6, x7, x8, x9, x10], dim=1)
        del x_ms_20, fusion_signal_20, x_80m_to_20m, x_160m_to_20m

        # 60m branch output
        x_ms_60 = torch.cat([x_240m_to_60m, x_480m_to_60m, x_1440m_to_60m], dim=1)
        output_60 = self.rrdb_block_60_fusion(fusion_signal_60, x_ms_60)
        GT_out_60 = torch.cat([x11, x12], dim=1)
        del x_ms_60, fusion_signal_60, x_240m_to_60m, x_480m_to_60m, x_1440m_to_60m

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
