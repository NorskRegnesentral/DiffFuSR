import os
from typing import Optional

import numpy as np
import tifffile as tiff
import torch
from PIL import Image
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score


def gaussian_downsample(tensor, scale_factor=0.25, kernel_size=8, sigma=3.0):
    """
    Apply Gaussian blur followed by downsampling
    Args:
        tensor: Input tensor (B, C, H, W)
        scale_factor: Downsampling factor
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation for Gaussian kernel
    """
    B, C, H, W = tensor.shape
    
    # Create Gaussian kernel
    x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
    kernel = torch.exp(-x**2 / (2*sigma**2))
    kernel = kernel / kernel.sum()
    
    # Create 2D kernel
    kernel = torch.outer(kernel, kernel)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(C, 1, 1, 1)
    
    # Apply padding
    pad_size = kernel_size//2
    tensor_padded = torch.nn.functional.pad(tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    # Apply Gaussian blur
    blurred = torch.nn.functional.conv2d(
        tensor_padded, 
        kernel.to(tensor.device), 
        groups=C,
        padding=0
    )
    
    # Calculate target size exactly
    target_h = int(round(H * scale_factor))
    target_w = int(round(W * scale_factor))
    
    # Downsample using area interpolation with exact size
    return torch.nn.functional.interpolate(
        blurred[:, :, :H, :W],  # Crop back to original size before scaling
        size=(target_h, target_w),
        mode='area'
    )

def read_tiff(
    path: str, channels: int = 12 , mode: Optional[str] = "RGB", to_float: bool = False
) -> np.ndarray:

    """Read a 12-channel TIFF image and return it as a numpy array"""

    img = tiff.imread(path)
    if channels == 3:
        img = img[:, :, [3,2,1]]
    if channels == 1: # unsqueeze the channel dimension
        img = np.expand_dims(img, axis=2)
        print(img.shape)
    if to_float:
        img = img.astype(float)

    return img

def calculate_ergas(img_lr, img_hr, scale=4):
    """
    Calculate ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse) metric
    Args:
        img_lr: Low-resolution image (H, W, C)
        img_hr: High-resolution image (H, W, C)
        scale: Resolution ratio between MS and PAN
    Returns:
        ERGAS value
    """
    img_lr = img_lr.astype(np.float32)
    img_hr = img_hr.astype(np.float32)
    
    num_channels = img_lr.shape[2]
    mse_per_band = np.zeros(num_channels)
    mean_per_band = np.zeros(num_channels)
    
    for i in range(num_channels):
        mse_per_band[i] = np.mean((img_lr[:,:,i] - img_hr[:,:,i])**2)
        mean_per_band[i] = np.mean(img_hr[:,:,i])
        
    rmse_per_band = np.sqrt(mse_per_band)
    relative_rmse = (rmse_per_band / (mean_per_band + np.finfo(float).eps))**2
    
    ergas = 100.0 * scale * np.sqrt(np.mean(relative_rmse))
    return ergas

def calculate_metrics(img1, img2):
    # Reshape images to 1D for some calculations
    img1_1d = img1.ravel()
    img2_1d = img2.ravel()

    # R2 Score
    r2 = r2_score(img1_1d, img2_1d)
    # print(f'R2 Score: {r2}')

    # Cross-Correlation
    cross_corr = np.correlate(img1_1d, img2_1d)
    norm_factor = np.sqrt((img1_1d ** 2).sum() * (img2_1d ** 2).sum())
    cross_corr = cross_corr / norm_factor

    # print(f'Cross-Correlation: {cross_corr}')

    # Calculate metrics for each channel separately and average results
    ssim_values = []
    psnr_values = []
    mse_values = []
    data_range = 255 #if img1.dtype == np.uint8 else 1.0
    #print(f'data_range: {data_range}')
    for ch in range(img1.shape[2]):
        # ssim_values.append(ssim(img1[:,:,ch], img2[:,:,ch], data_range=img2[:,:,ch].max() - img2[:,:,ch].min()))
        # psnr_values.append(psnr(img1[:,:,ch], img2[:,:,ch], data_range=img2[:,:,ch].max() - img2[:,:,ch].min()))
        ssim_values.append(ssim(img1[:,:,ch], img2[:,:,ch], data_range=data_range))
        psnr_values.append(psnr(img1[:,:,ch], img2[:,:,ch], data_range=data_range))
        mse_values.append(mse(img1[:,:,ch], img2[:,:,ch]))

    # Average over all channels
    avg_ssim = np.mean(ssim_values)
    avg_psnr = np.mean(psnr_values)
    avg_mse = np.mean(mse_values)

    # Calculate ERGAS metric
    ergas = calculate_ergas(img1, img2, scale=4)


    # print(f'Average SSIM: {avg_ssim}')
    # print(f'Average PSNR: {avg_psnr}')
    # print(f'Average MSE: {avg_mse}')
    return r2, cross_corr, avg_ssim, avg_psnr, avg_mse, ergas



# Get a list of all files in the directory
directory = 'logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/lr_sample/worldstrat/'# '/nr/bamjo/projects/SuperAI/data/jodata9_backup/paper2_eval_no_stretch/C12/LR/' #/nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/data
save_png = False

# Define the paths
paths_tiff = []

files = os.listdir(directory)
# Initialize variables to store total metrics
total_r2_1 = 0
total_cross_corr_1 = 0
total_ssim_1 = 0
total_psnr_1 = 0
total_mse_1 = 0
total_ergas_1 = 0

total_r2_2 = 0
total_cross_corr_2 = 0
total_ssim_2 = 0
total_psnr_2 = 0
total_mse_2 = 0
total_ergas_2 = 0

total_r2_3 = 0
total_cross_corr_3 = 0
total_ssim_3 = 0
total_psnr_3 = 0
total_mse_3 = 0
total_ergas_3 = 0

num_files = 0
type_model = 'worldstrat' # naip_harm, unharm, worldstrat
# Process each file
for file in files:
    # Only process .tiff files
    if file.endswith('.tif'):
        # Create the full file path by joining the directory and the file name
        file_path = os.path.join(directory, file)

        # Read and process the image /nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/data
        img_lr = np.transpose(read_tiff(file_path, channels=12), (1,2,0))
        img1 = np.transpose(read_tiff(file_path, channels=12), (1,2,0))#np.transpose(read_tiff('/nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/logs/blindsrsnf_aniso_jif_igars_3/version_3/lr_sample/worldstrat/' + file, channels=12), (1,2,0)) #read_tiff('/nr/bamjo/projects/SuperAI/data/jodata9_backup/paper2_eval_no_stretch/C1/SR_12/' + file, channels=12)
        if type_model == 'naip_harm':
            img2 = np.transpose(read_tiff('logs/blindsrsnf_aniso_naip_degraded_harm_large/version_1/results_opensr_12_NN_fuse/worldstrat/' + file, channels=12), (1,2,0)) #read_tiff('/nr/bamjo/projects/SuperAI/data/jodata9_backup/paper2_eval_no_stretch/C3/SR_12/' + file, channels=12)
            img3 = np.transpose(read_tiff('logs/blindsrsnf_aniso_naip_degraded_harm_large/version_1/results_opensr_12_GS_fuse/worldstrat/' + file, channels=12), (1,2,0))
        elif type_model == 'unharm':
            img2 = np.transpose(read_tiff('logs/blindsrsnf_aniso_naip_degraded_not_harm_large/version_0/results_opensr_12_NN_fuse/worldstrat/' + file, channels=12), (1,2,0))
            img3 = np.transpose(read_tiff('logs/blindsrsnf_aniso_naip_degraded_not_harm_large/version_0/results_opensr_12_GS_fuse/worldstrat/' + file, channels=12), (1,2,0))
        elif type_model == 'worldstrat':
            img2 = np.transpose(read_tiff('logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/results_opensr_12_NN_fuse/worldstrat/' + file, channels=12), (1,2,0))
            img3 = np.transpose(read_tiff('logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/results_opensr_12_GS_fuse/worldstrat/' + file, channels=12), (1,2,0))
        if save_png:
            Image.fromarray(img_lr[:,:,[3,2,1]].astype(np.uint8)).save(os.path.join(paths_tiff[0], file.replace('.tiff', '.png')))
            Image.fromarray(img_lr[:,:,[3,2,1]].astype(np.uint8)).save(os.path.join(paths_tiff[1], file.replace('.tiff', '.png')))
            Image.fromarray(img_lr[:,:,[3,2,1]].astype(np.uint8)).save(os.path.join(paths_tiff[2], file.replace('.tiff', '.png')))

        # Image.fromarray(img1[:,:,[3,2,1]].astype(np.uint8)).save(os.path.join(paths_tiff[3], file.replace('.tiff', '.png')))
        # Image.fromarray(img2[:,:,[3,2,1]].astype(np.uint8)).save(os.path.join(paths_tiff[4], file.replace('.tiff', '.png')))
        # Image.fromarray(img3[:,:,[3,2,1]].astype(np.uint8)).save(os.path.join(paths_tiff[5], file.replace('.tiff', '.png')))

        
        # Convert numpy arrays to torch tensors and rearrange dimensions to (batch, channel, height, width)
        tensor1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
        tensor2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
        tensor3 = torch.from_numpy(img3).permute(2, 0, 1).unsqueeze(0).float()
        tensorlr = torch.from_numpy(img_lr).permute(2, 0, 1).unsqueeze(0).float()

        # Downsample the tensors using torch.nn.functional.interpolate
        downsampled_tensor1 = torch.nn.functional.interpolate(tensor1, scale_factor=1, mode='bilinear', align_corners=False)
        #downsampled_tensor2 = torch.nn.functional.interpolate(tensor2, scale_factor=0.25, mode='bilinear', align_corners=False)
        # downsampled_tensor3 = torch.nn.functional.interpolate(tensor3, scale_factor=0.25, mode='bilinear', align_corners=False)

        # Replace the existing downsampling code with:
        #downsampled_tensor1 = gaussian_downsample(tensor1, scale_factor=1)
        downsampled_tensor2 = gaussian_downsample(tensor2, scale_factor=0.25)
        downsampled_tensor3 = gaussian_downsample(tensor3, scale_factor=0.25)

        # downsampled_tensor1 = downsample_by_resolution(tensor1)
        # downsampled_tensor2 = downsample_by_resolution(tensor2)
        # downsampled_tensor3 = downsample_by_resolution(tensor3)

        # Convert downsampled tensors back to numpy and rearrange dimensions back to (height, width, channel)
        img1 = downsampled_tensor1.squeeze(0).permute(1, 2, 0).numpy()
        img2 = downsampled_tensor2.squeeze(0).permute(1, 2, 0).numpy()
        img3 = downsampled_tensor3.squeeze(0).permute(1, 2, 0).numpy()
        img_lr = tensorlr.squeeze(0).permute(1, 2, 0).numpy()

        # Calculate metrics
        r2_1, cross_corr_1, avg_ssim_1, avg_psnr_1, avg_mse_1, ergas_1 = calculate_metrics(img_lr, img1)
        r2_2, cross_corr_2, avg_ssim_2, avg_psnr_2, avg_mse_2, ergas_2  = calculate_metrics(img_lr, img2)
        r2_3, cross_corr_3, avg_ssim_3, avg_psnr_3, avg_mse_3, ergas_3  = calculate_metrics(img_lr, img3)

        # Add ERGAS to the totals
        total_ergas_1 += ergas_1
        total_ergas_2 += ergas_2
        total_ergas_3 += ergas_3


        total_r2_1 += r2_1
        total_cross_corr_1 += cross_corr_1
        total_ssim_1 += avg_ssim_1
        total_psnr_1 += avg_psnr_1
        total_mse_1 += avg_mse_1

        total_r2_2 += r2_2
        total_cross_corr_2 += cross_corr_2
        total_ssim_2 += avg_ssim_2
        total_psnr_2 += avg_psnr_2
        total_mse_2 += avg_mse_2

        total_r2_3 += r2_3
        total_cross_corr_3 += cross_corr_3
        total_ssim_3 += avg_ssim_3
        total_psnr_3 += avg_psnr_3
        total_mse_3 += avg_mse_3

        num_files += 1

# Calculate and print average metrics

print('LR vs LR')
print(f'Average R2 Score: {total_r2_1 / num_files}')
print(f'Average Cross-Correlation: {total_cross_corr_1 / num_files}')
print(f'Average SSIM: {total_ssim_1 / num_files}')
print(f'Average PSNR: {total_psnr_1 / num_files}')
print(f'Average MSE: {total_mse_1 / num_files}')
print(f'Average ERGAS: {total_ergas_1 / num_files}')

# Calculate and print average metrics
print('NN 12 vs LR')
print(f'Average R2 Score: {total_r2_2 / num_files}')
print(f'Average Cross-Correlation: {total_cross_corr_2 / num_files}')
print(f'Average SSIM: {total_ssim_2 / num_files}')
print(f'Average PSNR: {total_psnr_2 / num_files}')
print(f'Average MSE: {total_mse_2 / num_files}')
print(f'Average ERGAS: {total_ergas_2 / num_files}')



# Calculate and print average metrics
print('GS 12 vs LR')
print(f'Average R2 Score: {total_r2_3 / num_files}')
print(f'Average Cross-Correlation: {total_cross_corr_3 / num_files}')
print(f'Average SSIM: {total_ssim_3 / num_files}')
print(f'Average PSNR: {total_psnr_3 / num_files}')
print(f'Average MSE: {total_mse_3 / num_files}')
print(f'Average ERGAS: {total_ergas_3 / num_files}')





