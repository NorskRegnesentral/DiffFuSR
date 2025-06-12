import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
from torch.nn.functional import interpolate
from fm4cs.arch.rrdbnet_arch import RRDBNet
from fm4cs.arch.rrdbnet_fuse_arch import FusionRRDBNet
import torchvision.transforms as transforms
from torch import Tensor

from torchmetrics.image import ErrorRelativeGlobalDimensionlessSynthesis
ergas = ErrorRelativeGlobalDimensionlessSynthesis()  

# def mean_absolute_error(preds: Tensor, target: Tensor) -> Tensor:
#     """Compute mean absolute error."""
#     return torch.mean(torch.abs(preds - target))


# def _l1_ergas_compute(
#     preds: Tensor,
#     target: Tensor,
#     ratio: float,
#     epsilon: float = 1e-8,
# ) -> Tensor:
#     """Compute L1 ERGAS."""
#     if not preds.is_floating_point():
#         preds = preds.float()
#     if not target.is_floating_point():
#         target = target.float()

#     num_bands = preds.shape[1]
#     mae_sq_sum = 0.0
#     for k in range(num_bands):
#         mae_bk = mean_absolute_error(preds[:, k, ...], target[:, k, ...])
#         mu_k = target[:, k, ...].mean()
#         mae_sq_sum += (mae_bk**2) / (mu_k**2 + epsilon)

#     ergas = (100 / ratio) * torch.sqrt((1 / num_bands) * mae_sq_sum)
#     return ergas


# class L1ErgasLoss(torch.nn.Module):
#     """L1 ERGAS loss function."""

#     def __init__(self, ratio: float = 4, epsilon: float = 1e-8):
#         super().__init__()
#         self.ratio = ratio
#         self.epsilon = epsilon

#     def forward(self, preds: Tensor, target: Tensor) -> Tensor:
#         """Forward pass."""
#         return _l1_ergas_compute(preds, target, self.ratio, self.epsilon).mean()



# def interpolate(x, scale_factor, mode='bilinear'):
#     """Bilinear downscaling"""
#     return nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode)#, align_corners=True


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

# TODO test this
def multi_resolution_gram_schmidt(inputs, target_size):
    """
    Multi-resolution Gram-Schmidt pan-sharpening
    Args:
        inputs: Dictionary containing S2 bands
        target_size: Target size for final output (typically 10m resolution size)
    Returns:
        Combined pansharpened image at target resolution
    """
    def gram_schmidt_single(ms_img, pan_img):
        # Standard GS implementation for a single resolution
        if pan_img.shape[1] == 3:
            pan_img = 0.2989 * pan_img[:,0:1] + 0.5870 * pan_img[:,1:2] + 0.1140 * pan_img[:,2:3]
        
        if pan_img.shape[-2:] != ms_img.shape[-2:]:
            pan_img = nn.functional.interpolate(pan_img, size=ms_img.shape[-2:], mode='bilinear', align_corners=False)
        
        ms_mean = ms_img.mean(dim=(2, 3), keepdim=True)
        ms_centered = ms_img - ms_mean
        synthetic_pan = ms_img.mean(dim=1, keepdim=True)
        
        ms_centered_flat = ms_centered.flatten(start_dim=2)
        synthetic_pan_flat = synthetic_pan.flatten(start_dim=2)
        
        covariance = torch.bmm(ms_centered_flat, synthetic_pan_flat.transpose(1, 2))
        covariance = covariance / ms_centered_flat.shape[2]
        
        variance = synthetic_pan_flat.var(dim=2, keepdim=True, unbiased=False) + 1e-6
        coefficients = (covariance / variance).unsqueeze(-1)
        
        pansharpened = ms_centered + coefficients * (pan_img - synthetic_pan)
        pansharpened = pansharpened + ms_mean
        
        return pansharpened

    # Extract bands
    rgb = torch.cat([
        inputs['S2:Red'],
        inputs['S2:Green'],
        inputs['S2:Blue']
    ], dim=1)

    # Process 10m bands
    ms_10m = torch.cat([
        inputs['S2:Red'],
        inputs['S2:Green'],
        inputs['S2:Blue'],
        inputs['S2:NIR']
    ], dim=1)
    pan_10m = interpolate(rgb, scale_factor=4)
    gs_10m = gram_schmidt_single(ms_10m, pan_10m)

    # Process 20m bands (downsample by 2 first)
    ms_20m = torch.cat([
        inputs['S2:RE1'],
        inputs['S2:RE2'],
        inputs['S2:RE3'],
        inputs['S2:RE4'],
        inputs['S2:SWIR1'],
        inputs['S2:SWIR2']
    ], dim=1)
    ms_20m_down = interpolate(ms_20m, scale_factor=1/2, mode='bilinear')
    pan_20m = interpolate(rgb, scale_factor=2)
    gs_20m = gram_schmidt_single(ms_20m_down, pan_20m)
    gs_20m_up = interpolate(gs_20m, size=target_size, mode='bilinear')

    # Process 60m bands (downsample by 6 first)
    ms_60m = torch.cat([
        inputs['S2:CoastAerosal'],
        inputs['S2:WaterVapor']
    ], dim=1)
    ms_60m_down = interpolate(ms_60m, scale_factor=1/6, mode='bilinear')
    pan_60m = interpolate(rgb, scale_factor=2/3)
    gs_60m = gram_schmidt_single(ms_60m_down, pan_60m)
    gs_60m_up = interpolate(gs_60m, size=target_size, mode='bilinear')

    # Combine all resolutions
    combined = torch.cat([gs_10m, gs_20m_up, gs_60m_up], dim=1)
    
    # Clean up
    del rgb, ms_10m, pan_10m, ms_20m, ms_20m_down, pan_20m
    del ms_60m, ms_60m_down, pan_60m, gs_10m, gs_20m, gs_60m
    torch.cuda.empty_cache()

    return combined

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
    def __init__(self,mode='train',GSD='GSD',lr=1e-4):
        super(FusionNetwork, self).__init__()
        #self.hparams.lr = 1e-4

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.mode = mode
        self.lr = lr  # Store learning rate as instance variable


        self.rrdb_block_60_fusion = FusionRRDBNet(3, 12, 2, nf=64, nb=5, gc=32, use_adain=False)
        self.rrdb_block_20_fusion = FusionRRDBNet(3, 10, 6, nf=64, nb=5, gc=32, use_adain=False)
        self.rrdb_block_10_fusion = FusionRRDBNet(3, 4, 4, nf=64, nb=5, gc=32, use_adain=False)

        self.mode = mode
        #self.GSD = GSD
        self.l1_criterion = nn.L1Loss()


    def forward(self, inputs, fusion_signal=None):
        # Extract input bands
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

        # y1 = inputs['S3:Oa01_reflectance']
        # y2 = inputs['S3:Oa02_reflectance']
        # y3 = inputs['S3:Oa03_reflectance']
        # y4 = inputs['S3:Oa04_reflectance']
        # y5 = inputs['S3:Oa05_reflectance']
        # y6 = inputs['S3:Oa06_reflectance']
        # y7 = inputs['S3:Oa07_reflectance']
        # y8 = inputs['S3:Oa08_reflectance']
        # y9 = inputs['S3:Oa09_reflectance']
        # y10 = inputs['S3:Oa10_reflectance']
        # y11 = inputs['S3:Oa11_reflectance']
        # y12 = inputs['S3:Oa12_reflectance']
        # y13 = inputs['S3:Oa13_reflectance']
        # y14 = inputs['S3:Oa14_reflectance']
        # y15 = inputs['S3:Oa15_reflectance']
        # y16 = inputs['S3:Oa16_reflectance']
        # y17 = inputs['S3:Oa17_reflectance']
        # y18 = inputs['S3:Oa18_reflectance']
        # y19 = inputs['S3:Oa19_reflectance']
        # y20 = inputs['S3:Oa20_reflectance']
        # y21 = inputs['S3:Oa21_reflectance']

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
                new_fusion_signal = torch.cat([fusion_signal['R'], fusion_signal['G'], fusion_signal['B']], dim=1)
                del fusion_signal  # Delete the old fusion_signal
                fusion_signal = new_fusion_signal
                del new_fusion_signal

            # Use the same fusion signal for all branches
            fusion_signal_60 = fusion_signal_20 = fusion_signal_10 = fusion_signal

            # Interpolate input bands
            x_40m_to_10m = interpolate(torch.cat([x1, x2, x3, x4], dim=1), 4)
            x_80m_to_20m = interpolate(torch.cat([x1, x2, x3, x4], dim=1), 4)
            x_160m_to_20m = interpolate(torch.cat([x5, x6, x7, x8, x9, x10], dim=1), 8)
            x_240m_to_60m = interpolate(torch.cat([x1, x2, x3, x4], dim=1), 4)
            x_480m_to_60m = interpolate(torch.cat([x5, x6, x7, x8, x9, x10], dim=1), 8)
            x_1440m_to_60m = interpolate(torch.cat([x11, x12], dim=1), 24)

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
        opt_10m, opt_20m, opt_60m = self.optimizers()
        losses = []

        # Train 10m branch
        output_10, _, _, GT_out_10, _, _ = self(batch)  # Only get needed outputs
        opt_10m.zero_grad()
        loss_10 = self.l1_criterion(output_10, GT_out_10)
        self.manual_backward(loss_10)
        opt_10m.step()
        self.log("loss_10m", loss_10, prog_bar=True)
        losses.append(loss_10.detach())
        # Clear memory
        #el output_10, GT_out_10, loss_10
        #torch.cuda.empty_cache()

        # Train 20m branch  
        _, output_20, _, _, GT_out_20, _ = self(batch)  # Only get needed outputs
        opt_20m.zero_grad()
        loss_20 = self.l1_criterion(output_20, GT_out_20)
        self.manual_backward(loss_20)
        opt_20m.step()
        self.log("loss_20m", loss_20, prog_bar=True)
        losses.append(loss_20.detach())
        # Clear memory
        #del output_20, GT_out_20, loss_20
        #torch.cuda.empty_cache()

        # Train 60m branch
        _, _, output_60, _, _, GT_out_60 = self(batch)  # Only get needed outputs
        opt_60m.zero_grad()
        loss_60 = self.l1_criterion(output_60, GT_out_60)
        self.manual_backward(loss_60)
        opt_60m.step()
        self.log("loss_60m", loss_60, prog_bar=True)
        losses.append(loss_60.detach())
        # Clear memory
        #del output_60, GT_out_60, loss_60
        #torch.cuda.empty_cache()

        # Log combined loss for monitoring
        total_loss = sum(losses) / len(losses)
        self.log("loss_train", total_loss, prog_bar=True)
        del losses

        if batch_idx == 0:
            self.trainer.train_dataloader.dataset.epoch += 1
    # def training_step(self, batch, batch_idx):
    #     # Forward pass returns outputs and ground truth for all resolutions

    #     output_10, output_20, output_60, GT_out_10, GT_out_20, GT_out_60 = self(batch, fusion_signal=None)

    #     # Calculate combined L1 loss across all resolutions
    #     loss = (self.l1_criterion(output_10, GT_out_10) + 
    #             self.l1_criterion(output_20, GT_out_20) + 
    #             self.l1_criterion(output_60, GT_out_60))
        
    #     # Calculate L1 ERGAS loss for each resolution
    #     # l1_ergas_loss_10 = self.l1_ergas_criterion(output_10, GT_out_10)
    #     # l1_ergas_loss_20 = self.l1_ergas_criterion_20(output_20, GT_out_20)
    #     # l1_ergas_loss_60 = self.l1_ergas_criterion_60(output_60, GT_out_60)
    #     # Combine losses
    #     # loss = (l1_ergas_loss_10 + l1_ergas_loss_20 + l1_ergas_loss_60)

    #     self.log("loss_train", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
    #     if batch_idx == 0:
    #         self.trainer.train_dataloader.dataset.epoch += 1
    #     return loss
    
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
    
    # def validation_step(self, batch, batch_idx):
    #     with torch.no_grad():
    #         # Forward pass
    #         output_10, output_20, output_60, GT_out_10, GT_out_20, GT_out_60 = self(batch, fusion_signal=None)
            
    #         # Interpolate ground truth outputs
    #         GT_out_10 = interpolate(GT_out_10, 4)
    #         GT_out_20 = interpolate(GT_out_20, 8) 
    #         GT_out_60 = interpolate(GT_out_60, 24)

    #         # Calculate GS outputs
    #         x_rgb = torch.cat([batch['S2:Red'], batch['S2:Green'], batch['S2:Blue']], dim=1)
    #         pan_2_5m = interpolate(x_rgb, 4)  # Simulate 2.5m pan image
    #         del x_rgb  # Free memory
            
    #         gs_output_10 = gram_schmidt_fusion(GT_out_10, pan_2_5m)
    #         gs_output_20 = gram_schmidt_fusion(GT_out_20, pan_2_5m)
    #         gs_output_60 = gram_schmidt_fusion(GT_out_60, pan_2_5m)
    #         del pan_2_5m  # Free memory

    #         # Calculate metrics
    #         metric_10 = ergas(output_10, GT_out_10).round()
    #         metric_20 = ergas(output_20, GT_out_20).round()
    #         metric_60 = ergas(output_60, GT_out_60).round()
    #         combined_metric = (metric_10 + metric_20 + metric_60) / 3

    #         gs_metric_10 = ergas(gs_output_10, GT_out_10).round()
    #         gs_metric_20 = ergas(gs_output_20, GT_out_20).round()
    #         gs_metric_60 = ergas(gs_output_60, GT_out_60).round()
    #         gs_combined_metric = (gs_metric_10 + gs_metric_20 + gs_metric_60) / 3

    #         # Log all metrics
    #         metrics = {
    #             "metric_val_10m": metric_10.item(),
    #             "metric_val_20m": metric_20.item(),
    #             "metric_val_60m": metric_60.item(),
    #             "metric_val": combined_metric.item(),
    #             "gs_metric_val_10m": gs_metric_10.item(),
    #             "gs_metric_val_20m": gs_metric_20.item(),
    #             "gs_metric_val_60m": gs_metric_60.item(),
    #             "gs_metric_val": gs_combined_metric.item()
    #         }
            
    #         # Log metrics in batch
    #         for name, value in metrics.items():
    #             self.log(name, value, prog_bar=name=="metric_val", sync_dist=True, on_epoch=True)

    #         # Only process images for first batch
    #         print(batch_idx)
    #         # if batch_idx == 0:
    #         #     # Extract first 6 samples of each band
    #         #     bands = {
    #         #         'Red': batch['S2:Red'][:6],
    #         #         'Green': batch['S2:Green'][:6],
    #         #         'Blue': batch['S2:Blue'][:6],
    #         #         'RE1': batch['S2:RE1'][:6],
    #         #         'RE2': batch['S2:RE2'][:6],
    #         #         'RE3': batch['S2:RE3'][:6],
    #         #         'Coast': batch['S2:CoastAerosal'][:6],
    #         #         'Water': batch['S2:WaterVapor'][:6]
    #         #     }

    #         #     # Create and normalize visualization grids
    #         #     grids = {
    #         #         'rgb': torchvision.utils.make_grid(torch.cat([bands['Blue'], bands['Green'], bands['Red']], dim=1).cpu()),
    #         #         're': torchvision.utils.make_grid(torch.cat([bands['RE3'], bands['RE2'], bands['RE1']], dim=1).cpu()),
    #         #         'coast': torchvision.utils.make_grid(torch.cat([bands['Water'], bands['Water'], bands['Coast']], dim=1).cpu())
    #         #     }
    #         #     del bands  # Free memory

    #         #     # Normalize and log input images
    #         #     for name, grid in grids.items():
    #         #         norm_grid = (grid - grid.min()) / (grid.max() - grid.min())
    #         #         self.logger.experiment.add_image(f'Input/{name.upper()}', norm_grid, batch_idx)
    #         #         del grid, norm_grid  # Free memory
    #         #     del grids

    #         #     # Log output images
    #         #     outputs = {
    #         #         '10m': (output_10.cpu(), GT_out_10.cpu(), gs_output_10.cpu()),
    #         #         '20m': (output_20.cpu(), GT_out_20.cpu(), gs_output_20.cpu()),
    #         #         '60m': (output_60.cpu(), GT_out_60.cpu(), gs_output_60.cpu())
    #         #     }

    #         #     for res, (pred, gt, gs) in outputs.items():
    #         #         self.logger.experiment.add_image(f'Output/{res}/Pred', make_norm_grid(pred).cpu(), batch_idx)
    #         #         self.logger.experiment.add_image(f'Output/{res}/GT', make_norm_grid(gt).cpu(), batch_idx)
    #         #         self.logger.experiment.add_image(f'GS/{res}/Output', make_norm_grid(gs).cpu(), batch_idx)
    #         #         del pred, gt, gs
    #         #     del outputs

    #         #     # Log metric comparison
    #         #     metric_comparison = {
    #         #         'Model vs GS': {
    #         #             '10m': float(metric_10 - gs_metric_10),
    #         #             '20m': float(metric_20 - gs_metric_20),
    #         #             '60m': float(metric_60 - gs_metric_60),
    #         #             'Combined': float(combined_metric - gs_combined_metric)
    #         #         }
    #         #     }
    #         #     self.logger.experiment.add_text('Metrics/Comparison', str(metric_comparison), batch_idx)
    #         #     del metric_comparison
    #         #     #pass

    #         # Cleanup remaining tensors
    #         del metric_10, metric_20, metric_60, combined_metric, gs_metric_10, gs_metric_20, gs_metric_60, gs_combined_metric

    #         del output_10, output_20, output_60
    #         del GT_out_10, GT_out_20, GT_out_60
    #         del gs_output_10, gs_output_20, gs_output_60
            
    #         return 9  # Return fixed metric for validation 
    
    def predict_step(self, batch, batch_idx, fusion_signal=None):
        # Get outputs from all three branches
        output_10, output_20, output_60, GT_out_10, GT_out_20, GT_out_60 = self(batch, fusion_signal=fusion_signal)
        
        # Helper function to create normalized grid (reusing from training)
        print('hello')

        # Log all three resolutions
        self.logger.experiment.add_image('Predict/10m/Output', make_norm_grid(output_10), batch_idx)
        self.logger.experiment.add_image('Predict/10m/GT', make_norm_grid(GT_out_10), batch_idx)
        self.logger.experiment.add_image('Predict/20m/Output', make_norm_grid(output_20), batch_idx)
        self.logger.experiment.add_image('Predict/20m/GT', make_norm_grid(GT_out_20), batch_idx)
        self.logger.experiment.add_image('Predict/60m/Output', make_norm_grid(output_60), batch_idx)
        self.logger.experiment.add_image('Predict/60m/GT', make_norm_grid(GT_out_60), batch_idx)
        # Return all outputs for GSD mode
        return output_10, output_20, output_60, GT_out_10, GT_out_20, GT_out_60

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    #     return optimizer
    def configure_optimizers(self):
        # Create separate optimizers for each network
        opt_10m = torch.optim.Adam(self.rrdb_block_10_fusion.parameters(), lr=self.lr)
        opt_20m = torch.optim.Adam(self.rrdb_block_20_fusion.parameters(), lr=self.lr)
        opt_60m = torch.optim.Adam(self.rrdb_block_60_fusion.parameters(), lr=self.lr)
        
        return [opt_10m, opt_20m, opt_60m]