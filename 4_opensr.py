#@title **Function to download pre-computed SR images**
import pathlib
from typing import Optional

import opensr_test
import pandas as pd
import requests


def downloadSR(
    model_id: str,
    dataset_id: Optional[str] = None,
    huggingface_repo: Optional[str] = "https://huggingface.co/isp-uv-es/superIX/resolve/main"
) -> pathlib.Path:
    """ Donwload the SR model results from the SUPER IX repository.

    Args:
        model_id (str): The model id.
        huggingface_repo (str): The Hugging Face repository.

    Returns:
        pathlib.Path: The path to the downloaded model.
    """
    # Set the dataset
    if dataset_id is None:
        dataset_id = opensr_test.datasets

    if isinstance(dataset_id, str):
        dataset_id = [dataset_id]

    for db in dataset_id:
        # download & load metadata
        print(f"Downloading {db}")
        metadata = f"https://huggingface.co/datasets/isp-uv-es/opensr-test/resolve/main/100/{db}/{db}_metadata.csv"
        metadata_db = pd.read_csv(metadata)

        # Set the model output path
        model_outpath = pathlib.Path(f"{model_id}/results/SR/{db}/geotiff")
        model_outpath.mkdir(parents=True, exist_ok=True)

        for i, row in metadata_db.iterrows():
            # File to download
            hr_file = row["hr_file"]
            dataset_path = f"{huggingface_repo}/{model_id}/results/SR/{db}/geotiff/{hr_file}.tif"

            # Download the file
            with requests.get(dataset_path, stream=True) as r:
                r.raise_for_status()
                with open(model_outpath / hr_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

    return pathlib.Path(f"{model_id}")



# For this example, we download the pre-computed super-resolution (SR) images
# from the opensrmodel. The images are stored in GEOTIFF format. To run your
# own model, ensure that your images are saved in the same format.
#downloadSR(model_id="opensrmodel")



#@title **Function to compute metrics**
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import torch


def image_resize(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an image using bilinear interpolation.

    Args:
        image (np.ndarray): The image to resize.
        size (int): The size to resize the image to.

    Returns:
        np.ndarray: The resized image.
    """
    image = torch.from_numpy(image)
    image = torch.nn.functional.interpolate(
        image / 10000,
        size = size,
        mode = "bilinear",
        antialias = True
    ) * 10000
    return image.squeeze().numpy()

def run(
    model_id: str,
    sr_in_bands: list,name_csv:str,
    scale: int = 4,
    dataset_ids: Optional[str] = None,
    experiment: Optional[dict] = None,
    output_dir: Optional[str] = None,
    compute_plots: Optional[dict] = {
        "triplets": True,
        "summary": True,
        "tc": True,
        "histogram": True,
        "ternary": True
    }
) -> None:
    """Run an experiment with opensr-test.

    Args:
        model_id (str, optional): The model to experiment with.
        dataset_ids (list, optional): The datasets to evaluate the model on.
        experiment (dict, optional): The experiment configuration.
        scale (int, optional): The scale factor to resize the SR images, if
            needed.
    """

    # Global parameters
    results_by_dataset = []
    exp_object = opensr_test.Metrics(**experiment)
    dmetric = experiment["correctness_distance"]
    columns = ["model", "dataset", "reflectance", "spectral", "spatial", "synthesis", "ha_metric", "om_metric", "im_metric"]
    condition = (dmetric == "clip") or (dmetric == "lpips")
    df = pd.DataFrame(columns=columns)

    for dataset in dataset_ids:

        # Set the output directory
        if output_dir is None:
            output_dir = pathlib.Path(f"{model_id}/results/SR/{dataset}/figures/")
            output_dir.mkdir(parents=True, exist_ok=True)

        output_dir_triplets = output_dir / "triplets"
        output_dir_triplets.mkdir(parents=True, exist_ok=True)

        output_dir_summary = output_dir / f"summary_{dmetric}"
        output_dir_summary.mkdir(parents=True, exist_ok=True)

        output_dir_tc = output_dir / f"triplets_tc_{dmetric}"
        output_dir_tc.mkdir(parents=True, exist_ok=True)

        output_dir_histogram = output_dir / f"histogram_{dmetric}"
        output_dir_histogram.mkdir(parents=True, exist_ok=True)

        output_dir_ternary = output_dir / f"ternary_{dmetric}"
        output_dir_ternary.mkdir(parents=True, exist_ok=True)

        # Load RGB LR & HR data
        data = opensr_test.load(dataset)
        lr, hr = data["L2A"][:, [3, 2, 1]], data["HRharm"][:, 0:3]
        metadata = data["metadata"]

        # Load SR data
        sr_path = pathlib.Path(f"{model_id}/results/SR/{dataset}/geotiff")
        sr_files = list(sr_path.glob("*"))
        sr_files.sort()
        sr = np.stack([rxr.open_rasterio(file)[sr_in_bands] for file in sr_files])
        sr_scale = sr.shape[2] // lr.shape[2]
        hr_scale = hr.shape[2] // lr.shape[2]

        # Resize SR images if needed
        if sr_scale > hr_scale:
            sr = image_resize(sr, hr.shape[3])

        # Resize HR and LR images if needed
        if scale < 4:
            hr = image_resize(hr, lr.shape[3] * scale)
            sr = image_resize(sr, lr.shape[3] * scale)

        # Compute metrics
        for i in range(len(lr)):
            print(f"Processing {dataset}: image {i +1}/{len(lr) + 1}")
            # Load image by image
            lr_img, hr_img, sr_img = lr[i], hr[i], sr[i]
            lr_img = torch.from_numpy(lr_img) / 10000
            hr_img = torch.from_numpy(hr_img) / 10000
            sr_img = torch.from_numpy(sr_img) / 10000

            # Add metadata
            hr_name = metadata.loc[i, "hr_file"]

            if condition:
                # From reflectance to true color
                # IMPORTANT ESTIMATE PERCEPTUAL METRICS IN TRUE COLOR IMAGE
                # TO CONVERT FROM REFLECTANCE TO TRUE COLOR WE FOLLOW TCI PIPELINE
                # https://sentiwiki.copernicus.eu/web/s2-products
                lr_img = (lr_img * 3).clip(0, 1)
                hr_img = (hr_img * 3).clip(0, 1)
                sr_img = (sr_img * 3).clip(0, 1)

            # Execute the experiment
            results = exp_object.compute(lr_img.float(), sr_img.float(), hr_img.float())
            results["model"] = model_id
            results["dataset"] = dataset
            df.loc[len(df)] = results

            # Save the figures
            if compute_plots["triplets"]:
                fig, axs = exp_object.plot_triplets()
                fig.savefig(output_dir_triplets / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["summary"]:
                fig, axs = exp_object.plot_summary()
                fig.savefig(output_dir_summary / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["tc"]:
                fig, axs = exp_object.plot_tc()
                fig.savefig(output_dir_tc / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["histogram"]:
                fig, axs = exp_object.plot_histogram()
                fig.savefig(output_dir_histogram / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["ternary"]:
                fig, axs = exp_object.plot_ternary()
                fig.savefig(output_dir_ternary / f"{dataset}__{hr_name}.png")
                plt.close(fig)

    # Save the results
    df.to_csv(output_dir.parent / name_csv, index=False)

    return df



datasets = ["naip", "spot", "spain_crops", "spain_urban"]


experiment_01 = {"device": "cuda", "agg_method": "patch", "patch_size": 1, "correctness_distance": "nd", "border_mask": 64}
experiment_02 = {"device": "cuda", "agg_method": "patch", "patch_size": 16, "correctness_distance": "lpips", "border_mask": 64}
experiment_03 = {"device": "cuda", "agg_method": "patch", "patch_size": 16, "correctness_distance": "clip", "border_mask": 64}

path_diffsr = 'logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/results/worldstrat/diffsr'

results_1 = run(
    model_id=path_diffsr,
    sr_in_bands=[0, 1, 2],
    dataset_ids=datasets,
    experiment=experiment_01,name_csv="results_nd.csv",
    compute_plots = {
        "triplets": True,
        "summary": True,
        "tc": True,
        "histogram": True,
        "ternary": True
    }
)

results_1 = run(
    model_id=path_diffsr,
    sr_in_bands=[0, 1, 2],
    dataset_ids=datasets,
    experiment=experiment_02,name_csv="results_lpips.csv",
    compute_plots = {
        "triplets": True,
        "summary": True,
        "tc": True,
        "histogram": True,
        "ternary": True
    }
)


results_1 = run(
    model_id=path_diffsr,
    sr_in_bands=[0, 1, 2],
    dataset_ids=datasets,
    experiment=experiment_03,name_csv="results_clip.csv",
    compute_plots = {
        "triplets": True,
        "summary": True,
        "tc": True,
        "histogram": True,
        "ternary": True
    }
)


# results_1 = run(
#     model_id="diffsr_naip_unharm",
#     sr_in_bands=[0, 1, 2],
#     dataset_ids=datasets,
#     experiment=experiment_02,name_csv="results_lpips.csv",
#     compute_plots = {
#         "triplets": True,
#         "summary": True,
#         "tc": True,
#         "histogram": True,
#         "ternary": True
#     }
# )


# results_1 = run(
#     model_id="diffsr_naip_unharm",
#     sr_in_bands=[0, 1, 2],
#     dataset_ids=datasets,
#     experiment=experiment_03,name_csv="results_clip.csv",
#     compute_plots = {
#         "triplets": True,
#         "summary": True,
#         "tc": True,
#         "histogram": True,
#         "ternary": True
#     }
# )


# results_1 = run(
#     model_id="diffsr_vanilla",
#     sr_in_bands=[0, 1, 2],
#     dataset_ids=datasets,
#     experiment=experiment_02,name_csv="results_lpips.csv",
#     compute_plots = {
#         "triplets": True,
#         "summary": True,
#         "tc": True,
#         "histogram": True,
#         "ternary": True
#     }
# )


# results_1 = run(
#     model_id="diffsr_vanilla",
#     sr_in_bands=[0, 1, 2],
#     dataset_ids=datasets,
#     experiment=experiment_03,name_csv="results_clip.csv",
#     compute_plots = {
#         "triplets": True,
#         "summary": True,
#         "tc": True,
#         "histogram": True,
#         "ternary": True
#     }
# )



# results_1 = run(
#     model_id="diffsr_ws",
#     sr_in_bands=[0, 1, 2],
#     dataset_ids=datasets,
#     experiment=experiment_02,name_csv="results_lpips.csv",
#     compute_plots = {
#         "triplets": True,
#         "summary": True,
#         "tc": True,
#         "histogram": True,
#         "ternary": True
#     }
# )


# results_1 = run(
#     model_id="diffsr_ws",
#     sr_in_bands=[0, 1, 2],
#     dataset_ids=datasets,
#     experiment=experiment_03,name_csv="results_clip.csv",
#     compute_plots = {
#         "triplets": True,
#         "summary": True,
#         "tc": True,
#         "histogram": True,
#         "ternary": True
#     }
# )