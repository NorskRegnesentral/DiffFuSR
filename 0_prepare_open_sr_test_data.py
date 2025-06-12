

import os
import shutil
import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import numpy as np

# fro  World strat
S2_SN7_MEAN = torch.Tensor([
    0.0769, 0.0906, 0.1162, 0.1318, 0.1619, 0.2275, 0.2500, 
    0.2588, 0.2667, 0.2684, 0.2461, 0.1874
]).to(torch.float32)

S2_SN7_STD = torch.Tensor([
    0.1708, 0.1672, 0.1614, 0.1677, 0.1698, 0.1513, 0.1490,
    0.1515, 0.1461, 0.2365, 0.1472, 0.1418
]).to(torch.float32)

HR_MEAN = torch.Tensor([S2_SN7_MEAN[3], S2_SN7_MEAN[2], S2_SN7_MEAN[1]]).to(torch.float32)
HR_STD = torch.Tensor([S2_SN7_STD[3], S2_SN7_STD[2], S2_SN7_STD[1]]).to(torch.float32)

def normalize_lr_image(lr_img):
    """
    Normalize LR image using S2 statistics
    Args:
        lr_img: numpy array or tensor of shape (C, H, W) or (B, C, H, W)
    Returns:
        normalized tensor of shape (B, C, H, W)
    """
    if isinstance(lr_img, np.ndarray):
        lr_tensor = torch.from_numpy(lr_img).float()
        # Add batch dimension if needed
        if lr_tensor.dim() == 3:
            lr_tensor = lr_tensor.unsqueeze(0)
    else:
        lr_tensor = lr_img
        # Add batch dimension if needed
        if lr_tensor.dim() == 3:
            lr_tensor = lr_tensor.unsqueeze(0)
    
    # Normalize using S2 statistics
    lr_tensor = (lr_tensor - S2_SN7_MEAN.view(1, -1, 1, 1).to(lr_tensor.device)) / S2_SN7_STD.view(1, -1, 1, 1).to(lr_tensor.device)
    return lr_tensor

def normalize_hr_image(hr_img):
    """
    Normalize HR RGB image using corresponding S2 RGB statistics
    Args:
        hr_img: numpy array or tensor of shape (3, H, W) or (B, 3, H, W)
    Returns:
        normalized tensor of shape (B, 3, H, W)
    """
    # Get RGB means and stds (bands 4,3,2 in S2 order)
    rgb_mean = S2_SN7_MEAN[[3,2,1]]  # Red, Green, Blue
    rgb_std = S2_SN7_STD[[3,2,1]]    # Red, Green, Blue
    
    if isinstance(hr_img, np.ndarray):
        hr_tensor = torch.from_numpy(hr_img).float()
        # Add batch dimension if needed
        if hr_tensor.dim() == 3:
            hr_tensor = hr_tensor.unsqueeze(0)
    else:
        hr_tensor = hr_img
        # Add batch dimension if needed
        if hr_tensor.dim() == 3:
            hr_tensor = hr_tensor.unsqueeze(0)
    
    # Normalize using RGB statistics
    hr_tensor = (hr_tensor - rgb_mean.view(1, -1, 1, 1).to(hr_tensor.device)) / rgb_std.view(1, -1, 1, 1).to(hr_tensor.device)
    return hr_tensor

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
        pan_img = (pan_img[:,0:1] +  pan_img[:,1:2] +  pan_img[:,2:3])/3

    # Ensure spatial dimensions match
    if pan_img.shape[-2:] != ms_img.shape[-2:]:
        pan_img = nn.functional.interpolate(pan_img, size=ms_img.shape[-2:], 
                                         mode='bilinear', align_corners=False)
    
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

# Base paths
base_path = "load/opensrtest/100"
lr_path = os.path.join(base_path, "lr")
hr_path = os.path.join(base_path, "hr")
hr_pan_path = os.path.join(base_path, "hr_pan")

# Create directories if they don't exist
os.makedirs(lr_path, exist_ok=True)
os.makedirs(hr_path, exist_ok=True)
os.makedirs(hr_pan_path, exist_ok=True)

# Define the mapping rules for each dataset
mapping_rules = {
    "naip": {
        "lr_subfolder": "naip/L2A",
        "hr_path": "naip/hr_harmonized",
        "lr_pattern": r"ROI_(\d+)__.*?T\d+_.*?\.tif",
        "hr_pattern": r"HR__ROI_\1__m_.*?\.tif"
    },
    "spain_crops": {
        "lr_subfolder": "spain_crops/L2A",
        "hr_path": "spain_crops/hr_harmonized",
        "lr_pattern": r"LR__ROI_(\d+)__.*?\.tif",
        "hr_pattern": r"HR__ROI_\1__PNOA.*?\.tif"
    },
    "spain_urban": {
        "lr_subfolder": "spain_urban/L2A",
        "hr_path": "spain_urban/hr_harmonized",
        "lr_pattern": r"LR__ROI_(\d+)__.*?\.tif",
        "hr_pattern": r"HR__ROI_\1__PNOA.*?\.tif"
    },
    "spot": {
        "lr_subfolder": "spot/L2A",
        "hr_path": "spot/hr_harmonized",
        "lr_pattern": r"ROI_(\d+)__.*?\.tif",
        "hr_pattern": r"ROI_\1__IMG_SPOT.*?\.tif"
    },
    "venus": {
        "lr_subfolder": "venus/L2A",
        "hr_path": "venus/hr_harmonized",
        "lr_pattern": r"ROI_(\d+)__.*?T\d+_.*?\.tif",
        "hr_pattern": r"ROI_\1\.tif"
    }
}


def copy_lr_files_from_subfolders():
    """Copy all LR files from subfolders to the main lr directory"""
    print("Copying LR files from subfolders...")
    copied_count = 0
    
    # Read the lr_files_list.txt if it exists
    lr_files_list_path = "lr_files_list.txt"
    if os.path.exists(lr_files_list_path):
        print(f"Reading file list from {lr_files_list_path}")
        with open(lr_files_list_path, 'r') as f:
            lr_files_to_copy = [line.strip() for line in f if line.strip()]
    else:
        print("lr_files_list.txt not found, copying all LR files from subfolders")
        lr_files_to_copy = None
    
    for dataset, rules in mapping_rules.items():
        lr_subfolder_path = os.path.join(base_path, rules["lr_subfolder"])
        
        if not os.path.exists(lr_subfolder_path):
            print(f"Warning: Subfolder {lr_subfolder_path} does not exist")
            continue
            
        print(f"Processing {dataset} subfolder...")
        
        for file in os.listdir(lr_subfolder_path):
            if file.endswith('.tif'):
                # If we have a specific list, check if this file should be copied
                if lr_files_to_copy is not None and file not in lr_files_to_copy:
                    continue
                    
                source_path = os.path.join(lr_subfolder_path, file)
                dest_path = os.path.join(lr_path, file)
                
                # Copy the file
                shutil.copy2(source_path, dest_path)
                print(f"  Copied {file} from {dataset}")
                copied_count += 1
    
    print(f"Total LR files copied: {copied_count}")
    return copied_count

def find_matching_hr(lr_file, dataset):
    """Find matching HR file based on ROI number"""
    rule = mapping_rules[dataset]
    lr_match = re.match(rule["lr_pattern"], os.path.basename(lr_file))
    if not lr_match:
        return None
        
    roi_num = lr_match.group(1)
    hr_dir = os.path.join(base_path, rule["hr_path"])
    
    if not os.path.exists(hr_dir):
        return None
    
    for hr_file in os.listdir(hr_dir):
        if re.match(rule["hr_pattern"].replace("\\1", roi_num), hr_file):
            return os.path.join(hr_dir, hr_file)
    
    return None

def main():
    # Step 1: Copy LR files from subfolders
    lr_files_copied = copy_lr_files_from_subfolders()
    
    if lr_files_copied == 0:
        print("No LR files were copied. Exiting.")
        return
    
    # Step 2: Copy matching HR files
    print("\nCopying matching HR files...")
    hr_copied_count = 0
    
    for lr_file in os.listdir(lr_path):
        if not lr_file.endswith('.tif'):
            continue
            
        lr_full_path = os.path.join(lr_path, lr_file)
        
        # Try each dataset's rules to find matching HR
        for dataset in mapping_rules.keys():
            matching_hr = find_matching_hr(lr_full_path, dataset)
            if matching_hr and os.path.exists(matching_hr):
                # Copy HR file to destination using LR filename
                hr_dest = os.path.join(hr_path, lr_file)
                shutil.copy2(matching_hr, hr_dest)
                print(f"  Copied {os.path.basename(matching_hr)} -> hr/{lr_file}")
                hr_copied_count += 1
                break
        else:
            print(f"  Warning: No matching HR file found for {lr_file}")
    
    print(f"Total HR files copied: {hr_copied_count}")
    
    # Step 3: Create pansharpened images
    print("\nCreating pansharpened images...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pansharpened_count = 0
    
    for lr_file in os.listdir(lr_path):
        if not lr_file.endswith('.tif'):
            continue
            
        lr_full_path = os.path.join(lr_path, lr_file)
        hr_full_path = os.path.join(hr_path, lr_file)
        
        if not os.path.exists(hr_full_path):
            print(f"  Warning: Missing HR pair for {lr_file}")
            continue
            
        try:
            # Read LR multispectral image
            with rasterio.open(lr_full_path) as src:
                lr_img = src.read()
                profile = src.profile
                # Convert to float32 and scale by 10000
                lr_img = lr_img.astype(np.float32) / 10000.0
            
            # Read HR RGB image
            with rasterio.open(hr_full_path) as src:
                hr_img = src.read()[:3]  # Only RGB bands
                # Convert to float32 and scale by 10000
                hr_img = hr_img.astype(np.float32) / 10000.0
            
            # Update profile for 4x upsampling
            profile.update({
                'height': profile['height'] * 4,
                'width': profile['width'] * 4,
                'transform': profile['transform'] * profile['transform'].scale(0.25, 0.25)
            })
            
            # Convert to torch tensors
            lr_tensor = torch.from_numpy(lr_img).float().unsqueeze(0).to(device)
            lr_tensor = F.interpolate(lr_tensor, size=(profile['height'], profile['width']), 
                                    mode='bilinear', align_corners=False)

            hr_tensor = torch.from_numpy(hr_img).float().unsqueeze(0).to(device)
            
            # Apply pansharpening
            with torch.no_grad():
                pansharpened = gram_schmidt_fusion(lr_tensor, hr_tensor)
                
            # Convert back to original scale
            pansharpened = pansharpened * S2_SN7_STD.view(1, -1, 1, 1).to(device) + S2_SN7_MEAN.view(1, -1, 1, 1).to(device)
            pansharpened = pansharpened.cpu().numpy()
            pansharpened = pansharpened.squeeze(0) * 10000.0  # Scale back to original range
            pansharpened = np.clip(pansharpened, 0, 65535).astype(np.uint16)  # Clip and convert to uint16
            
            # Save pansharpened image with original profile
            output_path = os.path.join(hr_pan_path, lr_file)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(pansharpened)
            
            print(f"  Created pansharpened version of {lr_file}")
            pansharpened_count += 1
            
        except Exception as e:
            print(f"  Error processing {lr_file}: {str(e)}")
    
    print(f"\nSummary:")
    print(f"  LR files copied: {lr_files_copied}")
    print(f"  HR files copied: {hr_copied_count}")
    print(f"  Pansharpened files created: {pansharpened_count}")

if __name__ == "__main__":
    main()