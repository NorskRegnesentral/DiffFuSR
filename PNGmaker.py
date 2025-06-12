

import os
import numpy as np
from PIL import Image
import tifffile as tiff

def read_tiff(path, channels=12):
    """Read a 12-channel TIFF image"""
    img = tiff.imread(path)
    return img

def create_rgb_visualization(img, band_combination, percentile=98):
    """
    Create RGB visualization from specified bands
    Args:
        img: Input image array (H,W,C) or (C,H,W)
        band_combination: List of 3 indices for R,G,B channels
        percentile: Percentile for contrast enhancement
    """
    if img.shape[0] == 12:  # If (C,H,W)
        img = np.transpose(img, (1,2,0))
    
    rgb = np.dstack([img[:,:,i] for i in band_combination])
    
    # Enhance contrast using percentile
    for i in range(3):
        p = np.percentile(rgb[:,:,i], percentile)
        rgb[:,:,i] = np.clip(rgb[:,:,i] / p * 255, 0, 255)
    
    return rgb.astype(np.uint8)

# Define interesting band combinations
BAND_COMBINATIONS = {
    'Natural_Color': [3,2,1],        # Red(output_10[0]), Green(output_10[1]), Blue(output_10[2])
    'False_Color': [7,3,2],          # NIR(output_10[3]), Red(output_10[0]), Green(output_10[1])
    'SWIR_Combo': [11,10,3],         # 12,11,4 SWIR2(output_20[5]), SWIR1(output_20[4]), Red(output_10[0])
    'Vegetation': [7,4,3],           # NIR(output_10[3]), RE1(output_20[0]), Red(output_10[0])
    'Urban_Features': [0,3,1],       # CoastAerosal(output_60[0]), Red(output_10[0]), Blue(output_10[2])
    'Agriculture': [8,7,3],          # RE4(output_20[3]), NIR(output_10[3]), Red(output_10[0])

    'Water_Penetration': [2,1,3],    # Blue(output_10[2]), Green(output_10[1]), Red(output_10[0])
    'Atmospheric': [9,9,1]           # WaterVapor(output_60[1]), CoastAerosal(output_60[0]), Green(output_10[1])
}


# Set up directories with separate NN and GS folders
base_dir = '' #/nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF
models = ['naip_harm', 'unharm', 'worldstrat']
methods = ['LR', 'GS', 'NN']
vis_dir = os.path.join(base_dir, 'visualizations_partial')

# Create visualization directories
os.makedirs(vis_dir, exist_ok=True)
for model in models:
    model_dir = os.path.join(vis_dir, model)
    os.makedirs(model_dir, exist_ok=True)
    for method in methods:
        method_dir = os.path.join(model_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        for combo_name in BAND_COMBINATIONS.keys():
            os.makedirs(os.path.join(method_dir, combo_name), exist_ok=True)


# Define list of names to process. If empty, process all images.
allowed_names = [
    'ROI_0006__20190725T054641_20190725T054643_T43TFF',
    'LR__ROI_00003__20210712T105619_20210712T110823_T30SWJ',
    'ROI_10903__20190715T172901_20190715T173906_T14UMU',
    'LR__ROI_00001__20210811T105619_20210811T110659_T30TXM',
    'ROI_0031__20200719T105031_20200719T105224_T31TCJ'

]

# Process images
for model in models:
    if model == 'naip_harm':
        lr_dir = os.path.join(base_dir, 'logs/blindsrsnf_aniso_naip_degraded_harm_large/version_1/lr_sample/worldstrat')
        hr_gs_dir = os.path.join(base_dir, 'logs/blindsrsnf_aniso_naip_degraded_harm_large/version_1/results_opensr_12_GS_fuse/worldstrat')
        hr_nn_dir = os.path.join(base_dir, 'logs/blindsrsnf_aniso_naip_degraded_harm_large/version_1/results_opensr_12_NN_fuse/worldstrat')
    elif model == 'unharm':
        lr_dir = os.path.join(base_dir, 'logs/blindsrsnf_aniso_naip_degraded_not_harm_large/version_0/lr_sample/worldstrat')
        hr_gs_dir = os.path.join(base_dir, 'logs/blindsrsnf_aniso_naip_degraded_not_harm_large/version_0/results_opensr_12_GS_fuse/worldstrat')
        hr_nn_dir = os.path.join(base_dir, 'logs/blindsrsnf_aniso_naip_degraded_not_harm_large/version_0/results_opensr_12_NN_fuse/worldstrat')
    else:  # worldstrat
        lr_dir = os.path.join(base_dir, 'logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/lr_sample/worldstrat')
        hr_gs_dir = os.path.join(base_dir, 'logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/results_opensr_12_GS_fuse/worldstrat')
        hr_nn_dir = os.path.join(base_dir, 'logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/results_opensr_12_NN_fuse/worldstrat')
    
    for file in os.listdir(lr_dir):
        if not file.endswith('.tif'):
            continue

        # Use the file's basename without the extension
        basename = file[:-4]
        # If allowed_names is not empty and the file is not in the list, skip it.
        if allowed_names and basename not in allowed_names:
            continue
            
        # Read LR and HR images
        lr_img = read_tiff(os.path.join(lr_dir, file))
        hr_gs_img = read_tiff(os.path.join(hr_gs_dir, file))
        hr_nn_img = read_tiff(os.path.join(hr_nn_dir, file))
        
        # Create visualizations for each band combination
        for combo_name, bands in BAND_COMBINATIONS.items():
            # Save LR visualization
            lr_rgb = create_rgb_visualization(lr_img, bands)
            Image.fromarray(lr_rgb).save(
                os.path.join(vis_dir, model, 'LR', combo_name, f'{basename}_{combo_name}.png'))
            
            # Save GS visualization
            hr_gs_rgb = create_rgb_visualization(hr_gs_img, bands)
            Image.fromarray(hr_gs_rgb).save(
                os.path.join(vis_dir, model, 'GS', combo_name, f'{basename}_{combo_name}.png'))
            
            # Save NN visualization
            hr_nn_rgb = create_rgb_visualization(hr_nn_img, bands)
            Image.fromarray(hr_nn_rgb).save(
                os.path.join(vis_dir, model, 'NN', combo_name, f'{basename}_{combo_name}.png'))
            
        print(f'Processed {file} for {model}')

print('Visualization complete! Check the visualizations directory.')


