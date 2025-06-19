import argparse
import os

import numpy as np
import torch
from litsr.data import PairedImageDataset, SingleImageDataset, DownsampledDataset
from litsr.metrics import calc_fid
from litsr.utils import mkdirs, read_yaml
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from tqdm import tqdm
import random
import tifffile
from models.fusion_model import FusionNetwork
from models import load_model
import numpy as np
import skimage
seed_everything(123)
import torch
import torch.nn as nn
from litsr.transforms import denormalize, normalize, tensor2uint8
import torch.nn.functional as F
import shutil

BAND_STATS = {
    "S2-10m": {
        "Red": {"mean": 0.1318, "std": 0.1677},     # Band 4
        "Green": {"mean": 0.1162, "std": 0.1614},   # Band 3
        "Blue": {"mean": 0.0906, "std": 0.1672},    # Band 2
        "NIR": {"mean": 0.2588, "std": 0.1515},     # Band 8
    },
    "S2-20m": {
        "RE1": {"mean": 0.1619, "std": 0.1698},     # Band 5
        "RE2": {"mean": 0.2275, "std": 0.1513},     # Band 6
        "RE3": {"mean": 0.2500, "std": 0.1490},     # Band 7
        "RE4": {"mean": 0.2667, "std": 0.1461},     # Band 8A
        "SWIR1": {"mean": 0.2461, "std": 0.1472},   # Band 11
        "SWIR2": {"mean": 0.1874, "std": 0.1418},   # Band 12
    },
    "S2-60m": {
        "CoastAerosal": {"mean": 0.0769, "std": 0.1708},  # Band 1
        "WaterVapor": {"mean": 0.2684, "std": 0.2365},    # Band 9
    }
}


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

def normalize_denormalize(x, mode='normalize', signal_type='fusion'):
    """
    Normalize or denormalize tensor based on Sentinel-2 statistics
    Args:
        x: Input tensor (B, C, H, W)
        mode: 'normalize' or 'denormalize'
        signal_type: 'fusion' (RGB only) or 'lr' (all 12 bands)
    Returns:
        Normalized/denormalized tensor
    """
    if signal_type == 'fusion':
        # For RGB fusion signal, use 10m Red, Green, Blue stats
        means = torch.tensor([
            BAND_STATS["S2-10m"]["Red"]["mean"],
            BAND_STATS["S2-10m"]["Green"]["mean"],
            BAND_STATS["S2-10m"]["Blue"]["mean"]
        ]).view(1, -1, 1, 1).to(x.device)
        
        stds = torch.tensor([
            BAND_STATS["S2-10m"]["Red"]["std"],
            BAND_STATS["S2-10m"]["Green"]["std"],
            BAND_STATS["S2-10m"]["Blue"]["std"]
        ]).view(1, -1, 1, 1).to(x.device)
    
    else:  # 'lr' - all 12 bands
        means = torch.tensor([
            BAND_STATS["S2-60m"]["CoastAerosal"]["mean"],
            BAND_STATS["S2-10m"]["Blue"]["mean"],
            BAND_STATS["S2-10m"]["Green"]["mean"],
            BAND_STATS["S2-10m"]["Red"]["mean"],
            BAND_STATS["S2-20m"]["RE1"]["mean"],
            BAND_STATS["S2-20m"]["RE2"]["mean"],
            BAND_STATS["S2-20m"]["RE3"]["mean"],
            BAND_STATS["S2-10m"]["NIR"]["mean"],
            BAND_STATS["S2-20m"]["RE4"]["mean"],
            BAND_STATS["S2-60m"]["WaterVapor"]["mean"],
            BAND_STATS["S2-20m"]["SWIR1"]["mean"],
            BAND_STATS["S2-20m"]["SWIR2"]["mean"]
        ]).view(1, -1, 1, 1).to(x.device)
        
        stds = torch.tensor([
            BAND_STATS["S2-60m"]["CoastAerosal"]["std"],
            BAND_STATS["S2-10m"]["Blue"]["std"],
            BAND_STATS["S2-10m"]["Green"]["std"],
            BAND_STATS["S2-10m"]["Red"]["std"],
            BAND_STATS["S2-20m"]["RE1"]["std"],
            BAND_STATS["S2-20m"]["RE2"]["std"],
            BAND_STATS["S2-20m"]["RE3"]["std"],
            BAND_STATS["S2-10m"]["NIR"]["std"],
            BAND_STATS["S2-20m"]["RE4"]["std"],
            BAND_STATS["S2-60m"]["WaterVapor"]["std"],
            BAND_STATS["S2-20m"]["SWIR1"]["std"],
            BAND_STATS["S2-20m"]["SWIR2"]["std"]
        ]).view(1, -1, 1, 1).to(x.device)

    if mode == 'normalize':
        return (x - means) / stds
    else:  # denormalize
        return x * stds + means

    #        lr_path="/nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/load/opensrtest/100/lr", multi_spectral=True,channels=12, # /nr/bamjo/user/msarmad/Work/SuperAI/BlindSRSNF/load/WorldStrat_dest_c3
       # hr_path="/nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/load/opensrtest/100/hr_pan",#
#
def make_dataloaders(scale, config):

        # Set channels explicitly for this script
    
    # Define the base path
    base_path = "load/opensrtest/100/lr"
    
    # Calculate the cache path that would be used
    bin_cache_path = os.path.join(
        os.path.dirname(base_path), 
        "_bin_" + os.path.basename(base_path)
    )
    
    # Check if bin cache exists and delete it if it does
    if os.path.exists(bin_cache_path):
        print(f"Removing existing cache: {bin_cache_path}")
        shutil.rmtree(bin_cache_path)

    dataset = PairedImageDataset(
        # lr_path="/nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/load/WorldStrat_c12_no_stretch/test/LR", multi_spectral=True,channels=12, # /nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/load/WorldStrat_c12_no_stretch/test/LR
        # hr_path="/nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/load/WorldStrat_c12_no_stretch/test/HR",#/nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/load/opensrtest/100/lr /nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/load/WorldStrat_c12_no_stretch/test/HR
        lr_path="/nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/load/opensrtest/100/lr_12", multi_spectral=True,channels=12, # /nr/bamjo/user/msarmad/Work/SuperAI/BlindSRSNF/load/WorldStrat_dest_c3
        hr_path="/nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/load/opensrtest/100/hr_pan",#
        scale=scale,
        is_train=False,
        cache="bin",
        rgb_range=config.data_module.args.get("rgb_range", 1),
        mean=config.data_module.args.get("mean"),
        std=config.data_module.args.get("std"),
        return_img_name=True,
    )
    print('Mean and Std:', config.data_module.args.get("mean"), config.data_module.args.get("std"))

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return loader


def test(args):
    # setup device
    if args.random_seed:
        seed_everything(random.randint(0, 1000))
    device = (
        torch.device("cuda", index=int(args.gpu)) if args.gpu else torch.device("cpu")
    )

    exp_path = os.path.dirname(os.path.dirname(args.checkpoint))
    ckpt_path = args.checkpoint

    # read config
    config = read_yaml(os.path.join(exp_path, "hparams.yaml"))

    # create model
    model = load_model(config, ckpt_path, strict=False)
    model.to(device)
    model.eval()

    model_fusion = FusionNetwork(mode='eval')   
    model_fusion.to(device)
    model_fusion.load_state_dict(torch.load('logs/GSD/lightning_logs/version_35/checkpoints/best-GSD-epoch=13-metric_val=4.695-loss_train=0.0329.ckpt')['state_dict'])
    #model_fusion.to(device)
    model_fusion.eval()


    scale = config.data_module.args.scale

    dataloader = make_dataloaders(scale=scale, config=config)

    # Set the result path based on the fusion method
    fusion_method = "gs" if args.use_gs else "nn"
    if args.use_gs:
        rslt_path = os.path.join(exp_path, "results_opensr_12_GS_fuse", "worldstrat")
    else:
        rslt_path = os.path.join(exp_path, "results_opensr_12_NN_fuse", "worldstrat")


    lr_path = os.path.join(exp_path, "lr_sample", "worldstrat") #rslt_path.replace("results", "lr_sample")
    mkdirs([rslt_path, lr_path])
    
    print(f"Using {fusion_method.upper()} fusion method")
    print(f"Results will be saved to: {rslt_path}")

    psnrs, ssims, run_times, losses = [], [], [], []
    for batch in tqdm(dataloader, total=len(dataloader.dataset)):
        lr, hr, name = batch # 1 x 12 x 160 x 160
        #batch = (lr.to(device), hr.to(device), name)
        # convert to 3 channels [3,2,1]
        batch = (lr[:, [3, 2, 1], :, :].to(device), hr[:, [3, 2, 1], :, :].to(device), name)

        # do test
        with torch.no_grad():
            rslt = model.test_step_lr_hr_paired(batch)
            # rslt["log_img_lr"] = model_fusion(rslt["log_img_lr"].unsqueeze(0).to(device)).squeeze(0)
            # rslt["log_img_sr"] = model_fusion(rslt["log_img_sr"].unsqueeze(0).to(device)).squeeze(0) 
            #fusion_signal = rslt["raw_img_sr"].to(device)


            # Select fusion method based on args.use_gs flag
            if not args.use_gs:
                # Neural Network (NN) fusion
                fusion_signal = normalize_denormalize(rslt["raw_img_sr"].to(device), 
                                                mode='normalize', 
                                                signal_type='fusion')
                lr_norm = normalize_denormalize(lr.to(device), 
                                          mode='normalize', 
                                          signal_type='lr')

                output_10, output_20, output_60, GT_out_10, GT_out_20, GT_out_60 = model_fusion(lr_norm.to(device), fusion_signal.to(device)) 
                fused_img = torch.cat([
                            output_60[:, 0:1],  # CoastAerosal from output_60 1
                            output_10[:, 2:3],  # Blue from output_10  2
                            output_10[:, 1:2],  # Green from output_10 3
                            output_10[:, 0:1],  # Red from output_10 4
                            output_20[:, 0:1],  # RE1 from output_20 5
                            output_20[:, 1:2],  # RE2 from output_20 6 
                            output_20[:, 2:3],  # RE3 from output_20 7
                            output_10[:, 3:4],  # NIR from output_10 8
                            output_20[:, 3:4],  # RE4 from output_20 9 # something wrong
                            output_60[:, 1:2],  # WaterVapor from output_60
                            output_20[:, 4:5],  # SWIR1 from output_20
                            output_20[:, 5:6]   # SWIR2 from output_20
                        ], dim=1)
                fused_img = normalize_denormalize(fused_img, mode='denormalize', signal_type='lr')

            else:
                # Gram-Schmidt (GS) fusion
                fusion_signal = rslt["raw_img_sr"].to(device)
                lr_upsampled = lr.to(device) #F.interpolate(lr.to(device), 
                                        #    scale_factor=4,  # or use size=(HR_height, HR_width)
                                        #    mode='bilinear', 
                                        #    align_corners=False)
                #fused_img = gram_schmidt_fusion(lr_upsampled, fusion_signal)
                fused_img = apply_gs_fusion_by_resolution(lr_upsampled, fusion_signal)
                
            # denormalize the fused image



            # output_10, output_20, output_60, GT_out_10, GT_out_20, GT_out_60
            # create a new tensor with the same shape as the original tensor
            fused_np = tensor2uint8([fused_img.cpu()[0]], model.rgb_range)
            rslt["log_img_sr"] = fused_np[0]
            rslt["log_img_lr"] = tensor2uint8([lr.cpu()[0]], model.rgb_range)[0]

        file_path = os.path.join(rslt_path, rslt["name"])
        lr_file_path = os.path.join(lr_path, rslt["name"])
        # print(rslt)
        # break
        if "log_img_sr" in rslt.keys():
            if isinstance(rslt["log_img_sr"], torch.Tensor):
                rslt["log_img_sr"] = rslt["log_img_sr"].cpu().numpy().transpose(1, 2, 0)
            tifffile.imwrite(file_path, rslt["log_img_sr"].transpose(2, 0, 1), planarconfig='separate')             #tifffile.imwrite(lr_file_path, rslt["log_img_lr"], planarconfig=PLANARCONFIG_SEPARATE)

            #skimage.io.imsave(file_path, rslt["log_img_sr"],photometric='minisblack')
            #plt.imsave(file_path, rslt["log_img_sr"])
        if "log_img_lr" in rslt.keys():
            if isinstance(rslt["log_img_lr"], torch.Tensor):
                rslt["log_img_lr"] = rslt["log_img_lr"].cpu().numpy().transpose(1, 2, 0)
            tifffile.imwrite(lr_file_path, rslt["log_img_lr"].transpose(2, 0, 1), planarconfig='separate')
            #skimage.io.imsave(lr_file_path, rslt["log_img_lr"],photometric='minisblack')

            #plt.imsave(lr_file_path, rslt["log_img_lr"])
        if "val_loss" in rslt.keys():
            losses.append(rslt["val_loss"])
        if "val_psnr" in rslt.keys():
            psnrs.append(rslt["val_psnr"])
        if "val_ssim" in rslt.keys():
            ssims.append(rslt["val_ssim"])
        if "time" in rslt.keys():
            run_times.append(rslt["time"])

    if losses:
        mean_loss = torch.stack(losses).mean()
        print("- Loss: {:.4f}".format(mean_loss))
    if psnrs:
        mean_psnr = np.array(psnrs).mean()
        print("- PSNR: {:.4f}".format(mean_psnr))
    if ssims:
        mean_ssim = np.array(ssims).mean()
        print("- SSIM: {:.4f}".format(mean_ssim))
    if run_times:
        mean_runtime = np.array(run_times[1:]).mean()
        print("- Runtime : {:.4f}".format(mean_runtime))

    #hr_path = "/nr/bamjo/user/msarmad/Work/SuperAI/BlindSRSNF/load/WorldStrat_c3/test/HR"
    #paths = [rslt_path, hr_path]
    #f3id_score = calc_fid(paths)
    #print("- FID : {:.4f}".format(fid_score))
    #print("=" * 42)


def getTestParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", default='logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/checkpoints/last.ckpt', type=str, help="checkpoint index") #logs/blindsrsnf_aniso_jif/version_0/checkpoints/last.ckpt
    parser.add_argument(
        "-g", "--gpu", default="3", type=str, help="indices of GPUs to enable"
    )
    parser.add_argument("--random_seed", action="store_true")
    parser.add_argument("--use_gs", default=True, action="store_true", help="Use Gram-Schmidt fusion instead of Neural Network fusion")

    return parser


test_parser = getTestParser()

if __name__ == "__main__":
    args = test_parser.parse_args()
    test(args)
