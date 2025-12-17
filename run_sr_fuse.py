#! /usr/bin/env python3

from argparse import ArgumentParser
import rasterio
from rasterio.windows import Window

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import dask.array
import numpy as np
import torch
import rioxarray
import xarray as xr

from litsr.utils import read_yaml
from litsr.utils.registry import ModelRegistry
from models.fusion_model import FusionNetwork
from utils.gram_schmidt_fusion import apply_gs_fusion_by_resolution
from utils.normalize_denormalize import normalize_denormalize


DEFAULT_CHECKPOINT = (
    "/nr/bamjo/projects/SuperAI/usr/sarmad/DiffFuSR/logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/checkpoints/last.ckpt"
)
DEFAULT_FUSION_CHECKPOINT = (
    "/nr/bamjo/projects/SuperAI/usr/sarmad/DiffFuSR/logs/GSD/lightning_logs/version_35/checkpoints/best-GSD-epoch=13-metric_val=4.695-loss_train=0.0329.ckpt"
)


def load_rs_model(
    checkpoint: Path,
    device: torch.device = torch.device("cpu")
) -> torch.nn.Module:
    """Load the RS model from a checkpoint."""
    exp_path = checkpoint.parent.parent
    config = read_yaml(exp_path / "hparams.yaml")
    model_name = config.lit_model.name
    model = ModelRegistry.get(model_name)
    model = model.load_from_checkpoint(
        checkpoint,
        opt=config,
        strict=False,
        map_location="cpu",
        weights_only=False,
    )
    model.to(device=device)
    model.eval()
    return model


def load_fusion_model(
    checkpoint: Path,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Load the fusion model."""
    model = FusionNetwork(mode="eval")
    model.load_state_dict(
        torch.load(checkpoint)["state_dict"]
    )
    model.to(device)
    model.eval()
    return model


def open_input(stack: ExitStack, file: Path) -> xr.Dataset:
    """Open the input file using the stack context."""
    if file.suffix.lower() in [".tif", ".tiff"]:
        da = stack.enter_context(
            rioxarray.open_rasterio(file, chunks="auto")
        )
        return da
    raise ValueError(f"Unsupported file format: {file.suffix}")


def init_output(da: xr.DataArray, res_multiplier=4, chunksize=640) -> xr.DataArray:
    """Initialize the output DataArray."""
    new_height = da.rio.height * res_multiplier
    new_width = da.rio.width * res_multiplier
    xres, yres = da.rio.resolution()
    new_xres = xres / res_multiplier
    new_yres = yres / res_multiplier
    xmin, ymin, xmax, ymax = da.rio.bounds()
    xcoords = np.linspace(xmin + new_xres / 2, xmax - new_xres / 2, new_width)
    # yres is negative and ycoords go from top to bottom
    ycoords = np.linspace(ymax + new_yres / 2, ymin - new_yres / 2, new_height)

    oda = xr.DataArray(
        data=dask.array.zeros(
            (da.rio.count, new_height, new_width),
            dtype=da.dtype,
            chunks=(da.rio.count, chunksize, chunksize)
        ),
        dims=da.dims,
        coords={
            "band": da.coords["band"],
            "y": ycoords,
            "x": xcoords,
        },
    )
    return oda


# def write_output(ofile, tfile: Path, crs=None):
#     """Write the output file from the temporary file."""
#     print(f"Writing output to {ofile}")
#     with xr.open_zarr(tfile, chunks="auto") as fused_ds:
#         fused_da = fused_ds["bands"]
#         if crs is not None:
#             fused_da.rio.write_crs(
#                 crs, inplace=True, grid_mapping_name="crs"
#             )
#         fused_da.rio.to_raster(
#             ofile, driver="GTiff", dtype="float32", compress="LZW"
#         )


def write_output(ofile, tfile: Path, crs=None, chunksize=2560):
    """Write the output file from the temporary file in chunks.

    rioxarray has issues writing large files from zarr in one go, so we
    need to do it manually using rasterio.
    """
    print(f"Writing output to {ofile}")
    with xr.open_zarr(tfile, chunks="auto") as fused_ds:
        fused_da = fused_ds["bands"]

        nbands, height, width = fused_da.shape

        # Get geospatial metadata
        xres = float(fused_da.x[1] - fused_da.x[0])
        yres = float(fused_da.y[1] - fused_da.y[0])
        xmin = float(fused_da.x[0]) - xres / 2
        ymax = float(fused_da.y[0]) - yres / 2

        transform = rasterio.transform.from_origin(xmin, ymax, xres, abs(yres))

        # Create output file
        profile = {
            'driver': 'COG',
            'dtype': 'float32',
            'count': nbands,
            'height': height,
            'width': width,
            'transform': transform,
            'compress': 'LZW',
            'tiled': True,
            'bigtiff': 'YES',
        }

        if crs is not None:
            profile['crs'] = crs

        with rasterio.open(ofile, 'w', **profile) as dst:
            # Write data in chunks
            for i in range(0, height, chunksize):
                for j in range(0, width, chunksize):
                    h = min(chunksize, height - i)
                    w = min(chunksize, width - j)

                    # Load chunk (this respects zarr chunking)
                    chunk = fused_da[:, i:i+h, j:j+w].values

                    # Write chunk for each band
                    window = Window(j, i, w, h)
                    for band_idx in range(nbands):
                        dst.write(chunk[band_idx], band_idx + 1, window=window)

                    print(f"Written chunk ({i}:{i+h}, {j}:{j+w})")

    print(f"Successfully wrote {ofile}")


def sr_fuse_image(
    im: torch.Tensor,
    model: torch.nn.Module,
    fusion_model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
):
    """Run SR Fusion on the input image."""

    if len(im.shape) == 3:
        im = im.unsqueeze(0)  # Add batch dimension
    batch = (
        im[:, [3,2,1]].to(device),
        [""],
    )
    with torch.no_grad():
        output = model.test_step_lr_only(batch)

        if fusion_model is None:
            # Use Gram-Schmidt fusion
            fusion_signal = output["sr_raw"].to(device)
            fused_image = apply_gs_fusion_by_resolution(
                im.to(device),
                fusion_signal
            )
            return fused_image

        # Use Neural Network fusion
        fusion_signal = normalize_denormalize(
            torch.tensor(output["sr_raw"]).to(device),
            mode="normalize",
            signal_type="fusion",
        )
        lr_norm = normalize_denormalize(
            im.to(device),
            mode="normalize",
            signal_type="lr",
        )
        output_10, output_20, output_60 = fusion_model(
            lr_norm.to(device), fusion_signal.to(device)
        )[:3]
        fused_image = torch.cat([
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
        fused_image = normalize_denormalize(
            fused_image,
            mode="denormalize",
            signal_type="lr",
        )
        return fused_image


def iter_slices(height: int, width: int, size: int):
    """Iterate over the image in slices."""
    bh, bw = size, size
    cnt = 0
    for i in range(0, height, bh):
        for j in range(0, width, bw):
            yield slice(i, min(i+bh, height)), slice(j, min(j+bw, width))
            cnt += 1
            if cnt>2:
                return


def sr_fuse(
    ofile: Path,
    file: Path,
    checkpoint: Path,
    fusion_checkpoint: Path,
    gram_schmidt: bool = False,
    device: int = None,
    patchsize: int = 160,
    tmpdir: Path = Path("/tmp")
):
    """Run SR Fusion."""
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", index=int(device))

    model = load_rs_model(checkpoint, device)
    model_fusion = None
    if not gram_schmidt:
        model_fusion = load_fusion_model(
            fusion_checkpoint or DEFAULT_CHECKPOINT,
            device=device,
        )

    with ExitStack() as stack:
        tdir = stack.enter_context(
            TemporaryDirectory(dir=tmpdir)
        )
        tfile = Path(tdir) / f"{file.name}.zarr"
        da = open_input(stack, file)
        oda = init_output(da)
        oda.to_dataset(name="bands").to_zarr(tfile)
        for ys, xs in iter_slices(da.rio.height, da.rio.width, patchsize):
            crop = torch.tensor(da[:, ys, xs].values, dtype=torch.float32)

            fused_crop = sr_fuse_image(
                crop,
                model,
                model_fusion,
                device=device,
            ).cpu().numpy()
            out_ys = slice(ys.start * 4, min(ys.stop * 4, oda.rio.height))
            out_xs = slice(xs.start * 4, min(xs.stop * 4, oda.rio.width))
            oda_win = oda[:, out_ys, out_xs]
            oda_win[:] = fused_crop[0]
            oda_win.to_dataset(name="bands").to_zarr(
                tfile,
                region={
                    "band": slice(None),
                    "y": out_ys,
                    "x": out_xs,
                },
                compute=True,
            )
        write_output(ofile, tfile, crs=da.rio.crs)


def main():
    """Parse args and run file."""
    parser = ArgumentParser(description="Run SuperAIPipe SR Fusion")
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="Model checkpoint file",
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Device to run the model on. Default is CPU.",
    )
    parser.add_argument(
        "--fusion-checkpoint", type=str, default=DEFAULT_FUSION_CHECKPOINT,
        help="Fusion model checkpoint file",
    )
    parser.add_argument(
        "--gram-schmidt", default=False, action="store_true",
        help="Use gram schmidt fusion instead of Neural Network fusion"
    )
    parser.add_argument(
        "--dask_procs", type=int, default=None,
    )
    parser.add_argument(
        "ofile", type=str, help="Output file to process",
    )
    parser.add_argument(
        "file", type=str, help="Input file to process",
    )
    args = parser.parse_args()
    if args.dask_procs is not None:
        dask.config.set(
            scheduler="threads",
            num_workers=args.dask_procs,
        )
    sr_fuse(
        ofile=Path(args.ofile),
        file=Path(args.file),
        checkpoint=Path(args.checkpoint),
        fusion_checkpoint=Path(args.fusion_checkpoint),
        gram_schmidt=args.gram_schmidt,
        device=args.device,
    )


if __name__ == "__main__":
    main()
