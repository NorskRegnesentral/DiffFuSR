import argparse
import os
import random
import shutil

import numpy as np
import rasterio
import tifffile
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

from litsr.data import SingleImageDataset
from litsr.utils import mkdirs, read_yaml
from models import load_model


def make_dataloaders(scale, config):
    # Set channels explicitly for this script
    channels = 3
    
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
    
    # Create the dataset with the correct channel configuration
    dataset = SingleImageDataset(
        img_path=base_path, 
        multi_spectral=True,
        channels=channels,  # Explicitly set to 3 channels
        cache="bin",        # This will now create a fresh cache
        rgb_range=config.data_module.args.get("rgb_range", 1),
        mean=config.data_module.args.get("mean"),
        std=config.data_module.args.get("std"),
        return_img_name=True,
        raw=True,
    )
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

    scale = config.data_module.args.scale

    dataloader = make_dataloaders(scale=scale, config=config)

    rslt_path = os.path.join(exp_path, "results", "worldstrat")

    lr_path = rslt_path.replace("results", "lr_sample")
    mkdirs([rslt_path, lr_path])

    psnrs, ssims, run_times, losses = [], [], [], []
    for batch in tqdm(dataloader, total=len(dataloader.dataset)):
        lr,  name = batch
        batch = (lr.to(device), name)
        # do test
        with torch.no_grad():
            rslt = model.test_step_lr_only(batch)

        file_path = os.path.join(rslt_path, rslt["name"])
        file_path_raw = os.path.join(rslt_path,'opensrtest', rslt["name"])
        if not os.path.exists(os.path.join(rslt_path,'opensrtest')):
            os.makedirs(os.path.join(rslt_path,'opensrtest'))
        lr_file_path = os.path.join(lr_path, rslt["name"])
        # print(rslt)
        # break
        if "log_img_sr" in rslt.keys():
            if isinstance(rslt["log_img_sr"], torch.Tensor):
                rslt["log_img_sr"] = rslt["log_img_sr"].cpu().numpy().transpose(1, 2, 0)
            #plt.imsave(file_path, rslt["log_img_sr"])
            tifffile.imwrite(file_path, rslt["log_img_sr"].transpose(2, 0, 1), planarconfig='separate')  
        if "sr_raw" in rslt.keys():

            rslt["sr_raw"] = ((rslt["log_img_sr"]/255)*10000).astype(np.uint16)



            # Define the metadata for the raster file
            metadata = {
                'driver': 'GTiff',
                'dtype': 'uint16',
                'nodata': None,
                'width': rslt["sr_raw"].shape[1],
                'height': rslt["sr_raw"].shape[0],
                'count': 3,  # number of bands
                'crs': None,
                #'transform': transform
            }

            # Write the data to a GeoTIFF file
            with rasterio.open(file_path_raw, 'w', **metadata) as dst:
                for i in range(rslt["sr_raw"].shape[2]):
                    dst.write(rslt["sr_raw"][:, :, i], i+1)  # Write each band to the file

            #tifffile.imwrite(file_path_raw, rslt["sr_raw"].transpose(2, 0, 1), planarconfig='separate')  
        if "log_img_lr" in rslt.keys():
            if isinstance(rslt["log_img_lr"], torch.Tensor):
                rslt["log_img_lr"] = rslt["log_img_lr"].cpu().numpy().transpose(1, 2, 0)
            #plt.imsave(lr_file_path, rslt["log_img_lr"])
            tifffile.imwrite(lr_file_path, rslt["log_img_lr"].transpose(2, 0, 1), planarconfig='separate')

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

   
def getTestParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", default='logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/checkpoints/last.ckpt', type=str, help="checkpoint index") #
    parser.add_argument(
        "-g", "--gpu", default="0", type=str, help="indices of GPUs to enable" # change to number 0 to n based on number of GPUs if needed.
    )
    parser.add_argument("--random_seed", action="store_true")

    return parser


test_parser = getTestParser()

if __name__ == "__main__":
    args = test_parser.parse_args()
    test(args)
