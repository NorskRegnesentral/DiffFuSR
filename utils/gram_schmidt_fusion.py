
import torch
import torch.nn.functional as F


def apply_gs_fusion_by_resolution(ms_img, pan_img):
    """
    Apply Gram-Schmidt fusion separately for 10m, 20m and 60m bands with appropriate downsampling
    Args:
        ms_img: Multispectral image tensor (B, 12, H, W)
        pan_img: RGB image tensor (B, 3, H, W)
    Returns:
        Fused image tensor (B, 12, H, W) with all bands
    """
    # Get original shapes
    Bp, _, Hp, Wp = pan_img.shape
    # Extract bands by resolution
    bands_10m = ms_img[:, [1,2,3,7], :, :]  # 4 bands
    bands_20m = ms_img[:, [4,5,6,8,10,11], :, :]  # 6 bands
    bands_60m = ms_img[:, [0,9], :, :]  # 2 bands

    # Downsample 20m and 60m bands
    H, W = ms_img.shape[2:]
    bands_20m = F.interpolate(bands_20m, size=(H//2, W//2), mode='bilinear', align_corners=False)
    bands_60m = F.interpolate(bands_60m, size=(H//6, W//6), mode='bilinear', align_corners=False)

    # Apply GS fusion to each resolution group
    fused_10m = gram_schmidt_fusion(bands_10m, pan_img)

    # Downsample pan image for 20m bands
    bands_20m = F.interpolate(bands_20m, size=(H//2, W//2), mode='bilinear', align_corners=False)
    fused_20m = gram_schmidt_fusion(bands_20m, pan_img)


    # Downsample pan image for 60m bands
    bands_60m = F.interpolate(bands_60m, size=(H//6, W//6), mode='bilinear', align_corners=False)
    fused_60m = gram_schmidt_fusion(bands_60m, pan_img)

    # Upsample fused 20m and 60m bands back to original resolution
    # fused_20m = F.interpolate(fused_20m, size=(H, W), mode='bilinear', align_corners=False)
    # fused_60m = F.interpolate(fused_60m, size=(H, W), mode='bilinear', align_corners=False)

    # Combine results back into single 12-band image
    fused_img = torch.zeros(Bp, 12, Hp, Wp, device=ms_img.device)

    # Place bands back in original positions
    fused_img[:, [1,2,3,7], :, :] = fused_10m
    fused_img[:, [4,5,6,8,10,11], :, :] = fused_20m
    fused_img[:, [0,9], :, :] = fused_60m

    return fused_img



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
    #ms_img = normalize_lr_image(ms_img)
    #pan_img = normalize_hr_image(pan_img)

    if pan_img.shape[1] == 3:
        #pan_img = 0.2989 * pan_img[:,0:1] + 0.5870 * pan_img[:,1:2] + 0.1140 * pan_img[:,2:3]
        pan_img = (pan_img[:,0:1] +  pan_img[:,1:2] +  pan_img[:,2:3])/3

    # Ensure spatial dimensions match
    # if pan_img.shape[-2:] != ms_img.shape[-2:]:
    #     pan_img = nn.functional.interpolate(pan_img, size=ms_img.shape[-2:],
    #                                      mode='bilinear', align_corners=False)

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

    # match the shape of , ms_centered and synthetic_pan to pan_img
    # Apply bilinear interpolation to ms_centered
    ms_centered = F.interpolate(ms_centered, size=pan_img.shape[-2:], mode='bilinear', align_corners=False)
    # Apply bilinear interpolation to synthetic_pan
    synthetic_pan = F.interpolate(synthetic_pan, size=pan_img.shape[-2:], mode='bilinear', align_corners=False)
    # Apply bilinear interpolation to coefficients
    coefficients = F.interpolate(coefficients, size=pan_img.shape[-2:], mode='bilinear', align_corners=False)

    # Apply GS transformation
    pansharpened = ms_centered + coefficients * (pan_img - synthetic_pan)
    pansharpened = pansharpened + ms_mean

    return pansharpened
