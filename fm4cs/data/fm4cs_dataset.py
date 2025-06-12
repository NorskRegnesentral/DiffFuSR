import math
import os
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from geopandas.tools._random import uniform
from pyproj import Transformer
from torch.utils.data import DataLoader, Dataset, get_worker_info

from fm4cs.core.dataset_registry import DATASETS
from fm4cs.data.fm4cs_dataset_base import FM4CSDatasetBase

# from fm4cs.data.utils import intersect_2_boxes, get_random_crop, print_raster

# from memory_profiler import profile
# mem_logs = open('mem_profile.log','a')


@DATASETS.register(override=True)
class FM4CSDataset(Dataset, FM4CSDatasetBase):
    def __init__(
        self,
        paths: str | Iterable[str],
        split: str = "train",
        products: list | dict = [  # noqa: B006
            "S2-10m",
            "S2-20m",
            "S2-60m",
            "S1-10m",
            "S1-60m",
        ],
        discard_bands: list = [],
        standardize: bool = True,
        minmax_normalize: bool = False,
        full_return: bool = False,
        img_size: int | None = None,  # Image size in pixels of the smallest GSD product
        ground_cover: float | None = None,
        data_percent: float = 1.0,
        max_nan_ratio: float = 0.05,
        pad: bool = False,
        resize=False,  # TODO: rename
        metadata_dir=None,
        include_filter: str | None = None,
        legacy_nan_handling: bool = False,
        # legacy_band_names: bool = True, # TODO: fix later, hard code to True for now
        sample_ratio: float = 1.0,
        custom_transform=None,
        **kwargs,
    ):
        super().__init__(
            paths=paths,
            split=split,
            products=products,
            discard_bands=discard_bands,
            img_size=img_size,
            ground_cover=ground_cover,
            data_percent=data_percent,
            max_nan_ratio=max_nan_ratio,
            include_filter=include_filter,
            legacy_band_names=True  # legacy_band_names, # TODO: set to False!
            ** kwargs,
        )

        self.standardize = standardize
        self.minmax_normalize = minmax_normalize
        self.full_return = full_return
        self.pad = pad  # TODO
        self.discard_bands = discard_bands
        self.resize = resize
        self.legacy_nan_handling = legacy_nan_handling
        self.sample_ratio = sample_ratio

        self.custom_transform = self._build_transforms(split == "train", self.products, self.product_changes)

        if metadata_dir is None:
            metadata_dir = f"{Path(self.paths)}_geometa"
        self.metadata_dir = Path(metadata_dir)

        self.metadata_cache_name = self.get_metadata_cache_name()

        self.file_paths, self.file_path_map = self.load_files(split, strict=True)
        print(f"Loaded {len(self.file_paths)} files")

        # DEBUG
        self._hits = 0
        self._bad_hit_dict = {"sum": 0}
        self.debug_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

    def _bad_hit(self, key):
        if key not in self._bad_hit_dict:
            self._bad_hit_dict[key] = 1
        else:
            self._bad_hit_dict[key] += 1
        self._bad_hit_dict["sum"] += 1

    def _print_bad_hits(self, total_hits):
        for key, value in self._bad_hit_dict.items():
            if key == "sum":
                continue
            print(f"{key}: {value / total_hits * 100:.2f}% [{value} / {total_hits}]")

    def __len__(self):
        # Using length of perfectly divisible square tiling of a tile
        # TODO: better solution
        num_tiles = (100_000 / self.ground_cover) ** 2  # 100km x 100km / (gc x gc)
        return int(len(self.file_paths) * num_tiles * self.sample_ratio)

    def _load_img(self, path, group=None):
        img_dataset = xr.open_dataset(
            path,
            group=group,
            # decode_coords="all",  # NOTE: h5netcdf does not like decode coords all
            cache=False,
            # chunks=None,
            # inline_array=True,
            engine="h5netcdf",  # NOTE: h5netcdf about 4x faster than netcdf4
        )

        return img_dataset

    @lru_cache(maxsize=1024)
    def _load_metadata(self, metadata_path):
        return gpd.read_file(metadata_path)

    def load_sample_areas(self, file_path):
        """
        Load all possible sample areas from the metadata file
        """

        fp = Path(file_path)
        rel_path = fp.relative_to(fp.parents[3])
        metadata_path = Path(self.metadata_dir) / rel_path / self.metadata_cache_name

        if metadata_path.exists():
            footprint_df = self._load_metadata(metadata_path)
            # shuffle the order of the metadata file to sample using the 'area' as weights
            footprint_df = footprint_df.sample(frac=1, weights="area", ignore_index=True)
            return footprint_df
        else:
            # print(f"Could not find metadata file at {metadata_path}")
            return None
            # raise FileNotFoundError(f"Could not find metadata file at {metadata_path}")

    # def __getitem__(self, idx):
    #     worker_info = get_worker_info()
    #     if worker_info is not None and worker_info.id == 0:
    #         local_rank = int(os.environ.get("LOCAL_RANK",0))
    #         print(f"local rank {local_rank} worker id {worker_info.id} getting item {idx}")
    #     product_imgs = {}
    #     for product in self.products:
    #         num_channels = len(self.NETCDF_PRODUCT_BAND_MAP[product])
    #         img_size = int( self.ground_cover / self.GSD(product))
    #         prod_img = np.random.randn(num_channels,img_size,img_size)
    #         product_imgs[product] = self.custom_transform[product](prod_img)
    #     return product_imgs

    # #@profile(stream=mem_logs)
    def __getitem__(self, idx):
        file_path = self.file_paths[idx % len(self.file_paths)]

        footprints_df = self.load_sample_areas(file_path)
        if footprints_df is None:
            self._bad_hit("no_footprint")
            return self.__getitem__(idx + 1)

        for sample_try in range(len(footprints_df)):
            found_all_products = True

            prod_imgs = footprints_df.iloc[sample_try].to_dict()

            product_imgs = {}
            crop_bounds = None
            crop_crs = None

            # NOTE: the order that we iterate through products needs to match
            #       the order that we initialized ConsistentRandomTransform so
            #       the correct scale is applied to each image
            for product in self.products:
                prod_img_dataset = self._load_img(
                    os.path.join(file_path, prod_imgs[product]),
                    product.split("-")[-1] if product not in self.SEPARATE_FOLDER_PRODUCTS else None,
                )

                if crop_bounds is None:
                    # Move to nearest pixel from largest GSD product
                    # TODO: if we resample the largest product, it might not be the largest product anymore!
                    # TODO: should we sort products largest to smallest by product_changes?
                    largest_prod = prod_img_dataset
                    largest_GSD = self.GSD(self.products[0])

                    crop_center_point = uniform(footprints_df.iloc[sample_try]["geometry"], 1, rng=None)  # TODO: rng

                    crop = (
                        crop_center_point.x - self.ground_cover / 2,  # TODO: should we subtract GSD/2 here?
                        crop_center_point.y - self.ground_cover / 2,
                    )

                    # pixel align
                    crop_min_xy = (
                        int(
                            largest_prod.x.sel(x=crop[0] + largest_GSD / 2, method="nearest").values.item()
                            - largest_GSD / 2
                        ),
                        int(
                            largest_prod.y.sel(y=crop[1] + largest_GSD / 2, method="nearest").values.item()
                            - largest_GSD / 2
                        ),
                    )

                    crop_bounds = (
                        crop_min_xy[0],
                        crop_min_xy[1],
                        crop_min_xy[0] + self.ground_cover,
                        crop_min_xy[1] + self.ground_cover,
                    )

                    crop_crs = prod_img_dataset.rio.crs
                    crop_crs_wkt = prod_img_dataset.crs.crs_wkt

                product_gsd = self.GSD(product)
                img_size = math.ceil(self.ground_cover / product_gsd)

                # Handle N and S UTMs difference
                if prod_img_dataset.crs.crs_wkt != crop_crs_wkt:  # TODO remove or keep?
                    transformer = Transformer.from_crs(crop_crs_wkt, prod_img_dataset.crs.crs_wkt, always_xy=True)

                    # Transform the coordinates of the bounding box
                    (x_min, y_min, x_max, y_max) = crop_bounds
                    x_min_transformed, y_min_transformed = transformer.transform(x_min, y_min)
                    x_max_transformed, y_max_transformed = transformer.transform(x_max, y_max)

                    prod_crop_bounds = (x_min_transformed, y_min_transformed, x_max_transformed, y_max_transformed)

                else:
                    prod_crop_bounds = crop_bounds

                # NOTE: possibly faster method, works nice with new bounds
                x = np.arange(prod_crop_bounds[0], prod_crop_bounds[2], self.GSD(product))
                y = np.arange(prod_crop_bounds[1], prod_crop_bounds[3], self.GSD(product))

                x += int(self.GSD(product) / 2)
                y += int(self.GSD(product) / 2)

                if y.max() > prod_img_dataset.y.max() or x.max() > prod_img_dataset.x.max():
                    # pdb.set_trace()
                    # TODO: debug why this happens
                    self._bad_hit("out_of_bounds")
                    found_all_products = False
                    break

                # #extract the indices of the image
                prod_img = prod_img_dataset.sel(x=x, y=y, method="nearest")
                # prod_img = prod_img_dataset.rio.clip_box(*crop_bounds, crs=crop_crs)

                if prod_img.sizes["x"] != img_size or prod_img.sizes["y"] != img_size:
                    self._bad_hit("size_mismatch")
                    found_all_products = False
                    break

                # Convert to numpy array
                prod_img = prod_img.bands.values

                nan_ratio = np.isnan(prod_img).sum(axis=(1, 2)) / (prod_img.shape[1] * prod_img.shape[2])
                if (nan_ratio > self.max_nan_ratio).any():
                    self._bad_hit("nan_ratio")
                    # self.debug_plot_footprints(footprint, file_path, prod_img_paths, crop_center_point, crop_bounds, crs_wkt=crop_crs_wkt, idx=idx, nan_ratio=nan_ratio)
                    found_all_products = False
                    break

                # if "S1" in product and not self.legacy_band_names: # TODO: this will not work for v1 dataset
                #     # Check which operation mode the S1 product is in
                #     #operation_mode = prod_img_dataset.source_meta.attrs["sensoroperationalmode"]
                #     if "1SDH" in prod_imgs[product]:
                #         polarization = "HH"
                #     else:
                #         polarization = "VV"
                #     #print("operation mode", operation_mode)
                #     #print("polarization", polarization)
                #     #print("prod imgs", prod_imgs[product])
                #     product_name, product_gsd = product.split("-")
                #     product_key = f"{product_name}-{polarization}-{product_gsd}"

                # else:
                product_key = product

                product_imgs[product_key] = prod_img

            if found_all_products:
                break

        if not found_all_products:
            # print(f"did not find a sample at idx {idx}! increasing idx and retrying...")
            return self.__getitem__(idx + 1)

        """
        ####### DEBUG #######
        crop_csv_file = Path("debug_plots/crop_bounds.csv")

        if not crop_csv_file.parent.exists():
            crop_csv_file.parent.mkdir(parents=True, exist_ok=True)

        if not crop_csv_file.exists():
            with open(crop_csv_file, "w") as f:
                f.write("file_path;s2_path;s1_path;crop_minx;crop_miny;crop_maxx;crop_maxy;footprint_minx;footprint_miny;footprint_maxx;footprint_maxy;crs\n")

        with open(crop_csv_file, "a") as f:
            f.write(f"{file_path};{prod_img_paths['S2']};{prod_img_paths['S1']};{crop_bounds[0]};{crop_bounds[1]};{crop_bounds[2]};{crop_bounds[3]};{footprint[0]};{footprint[1]};{footprint[2]};{footprint[3]};{crop_crs}\n")

        """

        # NOTE: the order that we iterate through products needs to match
        #       the order that we initialized ConsistentRandomTransform so
        #       the correct scale is applied to each image
        # consistent cropping in custom transforms has an internal counter,
        # only trigger counting once we are sure we have all the product images
        # NOTE: from here the product will be potentially resized
        for product, prod_img in product_imgs.items():
            # if "S1" in product:
            #    product_imgs[product] = self.custom_transform[self.lookup_product(product)](prod_img)
            # else: # TODO: fix legacy band names
            product_imgs[product] = self.custom_transform[product](prod_img)

        self._hits += 1
        if self._hits % 1000 == 0:
            worker_info = get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id

            if worker_id == 0:
                print(f"Worker {worker_id}, idx: {idx}, hits: {self._hits}")
                print(f"crop hit rate: {self._hits / (self._hits + self._bad_hit_dict['sum']) * 100:.2f}%")
                print("Bad hits: ")
                self._print_bad_hits(self._bad_hit_dict["sum"])
                # print(f"Cache info load img: {self._load_img.cache_info()}")
        if self.full_return:
            return product_imgs, idx
        return product_imgs

    def collate_fn(self, batch):
        if self.full_return:
            product_imgs, paths = zip(*batch, strict=False)
        else:
            all_product_imgs = batch

        collated_imgs = {}
        # TODO: just first stack products, then add band split option afterwards...
        for product in self.products:  # TODO: should we allow specifying only using for example S1 IW? # TOOD: legacy v1 dataset does not even handle IW and EW
            band_imgs = {}
            for product_imgs in all_product_imgs:
                product_img = product_imgs[product]

                # Need product name to differentiate between S1 multilooked to different resolutions
                for band_name, prod_band_name in zip(
                    self.NETCDF_PRODUCT_BAND_MAP[product].values(),
                    self.PRODUCT_BAND_NAME_MAP[product].values(),
                    strict=False,
                ):
                    prod_bands = self._get_product_bands(product, netcdf=True)
                    band_img = product_img[prod_bands.index(band_name)]

                    if band_name not in self.discard_bands:
                        # testing how usat works with nans # TODO add back
                        if (nan_index := torch.isnan(band_imgs)).any():
                            if self.legacy_nan_handling:
                                band_imgs[nan_index] = -0.1
                            else:
                                # TODO: median or mean?
                                band_mean = torch.nanmean(band_imgs, dim=(0, 1), keepdim=True).repeat(band_imgs.shape)
                                band_imgs[nan_index] = band_mean

                        if prod_band_name not in band_imgs:
                            band_imgs[prod_band_name] = []

                        band_imgs[prod_band_name].append(band_img[None, ...])

            for prod_band_name, imgs in band_imgs.items():
                if product != self.product_changes[product]:
                    # if "S1" in product:
                    #    product_name, _operation_mode, product_gsd = product.split("-")
                    # else: # TODO: fix legacy band names
                    product_name, product_gsd = product.split("-")
                    product_gsd = product_gsd.replace("m", "")
                    # S1 product has the GSD in the band name, need to update if we resample
                    if product_name == "S1":
                        prod_band_name = prod_band_name.replace(
                            product_gsd, str(self.GSD(self.product_changes[product]))
                        )

                collated_imgs[prod_band_name] = torch.stack(imgs)
                # print(f"product: {product}, band: {prod_band_name}, shape: {collated_imgs[prod_band_name].shape}")

        if self.full_return:
            return collated_imgs, paths
        return collated_imgs

    def plot(self, sample):
        B = sample[self._get_product_bands(self.products[0])[0]].shape[0]

        fig, ax = plt.subplots(B, len(self.products), figsize=(len(self.products) * 4, B * 4))

        for b in range(B):
            for i, product in enumerate(self.products):
                if product == "S2-10m":
                    bands = self.rgb_bands
                else:
                    # Take first, TODO: better idea
                    prod_bands = self._get_product_bands(product)
                    prod_band_name = prod_bands[0]
                    if product != self.product_changes[product]:
                        product_name, product_gsd = product.split("-")
                        product_gsd = product_gsd.replace("m", "")
                        # S1 product has the GSD in the band name, need to update if we resample
                        if product_name == "S1":
                            prod_band_name = prod_band_name.replace(
                                product_gsd, str(self.GSD(self.product_changes[product]))
                            )

                    bands = [prod_band_name]

                image = torch.stack([sample[band][b] for band in bands])

                image = torch.squeeze(image, dim=(1)).cpu().numpy()
                image = image.transpose(1, 2, 0)
                image = np.ma.masked_invalid(image)

                ax[b, i].imshow(image)
                ax[b, i].set_title(product)

        plt.show()

        return fig


if __name__ == "__main__":
    root = "/lokal-uten-backup-4tb/pro/fm4cs/usr/tforgaard/data/proc4/export5"

    dataset = FM4CSDataset(
        root,
        products={
            "S2-10m": None,
            "S2-20m": None,
            "S2-60m": None,  # "S2-50m",
            "S1-10m": None,
            # "S1-50m": None,
        },
        discard_bands=[],
        ground_cover=4800,
        standardize=True,
        data_percent=1.0,
        resize=True,
    )
    print(len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        num_workers=1,
        prefetch_factor=1,
    )

    for i_batch, sample_batched in enumerate(dataloader):
        print(sample_batched.keys())
        print(sample_batched[list(sample_batched.keys())[0]].shape)
        # fig = dataset.plot(sample_batched)
        # fig.savefig(f"test2_{i_batch}.png")
        # print(i_batch, sample_batched['image'].size())
        if i_batch >= 5:
            break
