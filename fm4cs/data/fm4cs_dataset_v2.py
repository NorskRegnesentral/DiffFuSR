import math
import os
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import xarray as xr
from geopandas.tools._random import uniform
from pyproj import Proj, Transformer
from torch.utils.data import Dataset, get_worker_info

from fm4cs.core.dataset_registry import DATASETS

# from fm4cs.data.utils import intersect_2_boxes, get_random_crop, print_raster
from fm4cs.data.fm4cs_dataset_base import FM4CSDatasetBase

# from memory_profiler import profile
# mem_logs = open('mem_profile.log','a')


@DATASETS.register(override=True)
class FM4CSDatasetV2(Dataset, FM4CSDatasetBase):
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
        minmax_normalize: bool = True,
        full_return: bool = False,
        img_size: int | None = None,  # Image size in pixels of the smallest GSD product
        ground_cover: float | None = None,
        data_percent: float = 1.0,
        max_nan_ratio: float = 0.05,
        pad: bool = False,  # TODO: remove
        resize: bool = False,  # TODO: rename
        metadata_dir: str | None = None,
        strict_data_mode: bool = False,
        include_filter: str | None = None,
        exclude_filter: str | None = None,
        legacy_nan_handling: bool = False,
        legacy_band_names: bool = False,
        sample_ratio: float = 1.0,
        batch_size: int = 1,
        random_seed: int = 42,
        custom_transform=None,  # TODO: remove?
        **kwargs,
    ):
        super().__init__(
            paths=paths,
            split=split,
            products=products,
            discard_bands=discard_bands,
            img_size=img_size,
            ground_cover=ground_cover,
            minmax_normalize=minmax_normalize,
            standardize=standardize,
            resize=resize,
            full_return=full_return,
            data_percent=data_percent,
            max_nan_ratio=max_nan_ratio,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            legacy_nan_handling=legacy_nan_handling,
            legacy_band_names=legacy_band_names,
            random_seed=random_seed,
            **kwargs,
        )

        self.strict_data_mode = strict_data_mode
        self.sample_ratio = sample_ratio
        self.batch_size = batch_size

        self.custom_transform = self._build_transforms(split == "train", self.products, self.product_changes)

        if metadata_dir is None:
            metadata_dir = self.get_metadata_dir()
        self.metadata_dir = Path(metadata_dir)

        out_products = sorted({self.SUBDIR_MAP[prod] for prod in self.products})
        out_filename = f"{'_'.join(out_products)}_{self.ground_cover}.geojson"
        out_filename = out_filename.replace("/", "_")
        self.metadata_cache_name = out_filename

        self.file_paths, _ = self.load_files(split, strict=False)  # or TODO: strict_data_mode
        print(f"Loaded {len(self.file_paths)} files")
        self.file_paths = np.array([str(fp) for fp in self.file_paths])
        print("file_path dtype", self.file_paths.dtype)

        # DEBUG
        self._hits = 0
        self._bad_hits = Counter()
        self.debug_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

    def _print_bad_hits(self):
        total_hits = self._bad_hits.total()
        for key, value in self._bad_hits.items():
            print(f"{key}: {value / total_hits * 100:.2f}% [{value} / {total_hits}]")

    def __len__(self):
        # Using length of perfectly divisible square tiling of a tile
        # TODO: better solution
        num_tiles = (100_000 / self.ground_cover) ** 2  # 100km x 100km / (gc x gc)
        return int(len(self.file_paths) * num_tiles * self.sample_ratio / self.batch_size)

    def _load_img(self, path, group=None):
        img_dataset = xr.open_dataset(
            path,
            group=group,
            # decode_coords="all",  # NOTE: h5netcdf does not like decode coords
            cache=False,
            # chunks=None,
            # inline_array=True,
            lock=False,
            engine="h5netcdf",  # NOTE: h5netcdf about 4x faster than netcdf4
        )

        return img_dataset

    @lru_cache(maxsize=2048)
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
            footprint_df = footprint_df.sample(
                frac=1, weights="area_prod", ignore_index=True, random_state=np.random.get_state()[1]
            )
            return footprint_df
        else:
            return None

    # @profile(stream=mem_logs)
    def __getitem__(self, idx):
        # worker_info = get_worker_info()
        # worker_id = worker_info.id if worker_info is not None else 0

        # Fetches one batch of data, from one single file
        batch_size = self.batch_size
        file_path = self.file_paths[idx % len(self.file_paths)]
        # print("idx", idx, "file", file_path , "worker_id", worker_id)

        footprints_df = self.load_sample_areas(file_path)

        if footprints_df is None:
            self._bad_hits["no_footprint"] += 1
            return self.__getitem__(idx + 1)

        for sample_try in range(len(footprints_df)):
            found_all_products = True
            product_imgs = {}
            crop_bounds = None
            crop_crs = None
            single_band_S1 = None
            skip_products = []

            prod_imgs = footprints_df.iloc[sample_try].to_dict()

            filtered_products = [p for p in self.products if p in prod_imgs and prod_imgs[p] is not None]

            if self.strict_data_mode and not all(any(s in p for s in self.sensors) for p in filtered_products):
                self._bad_hits["missing_products"] += 1
                continue

            # NOTE: the order that we iterate through products needs to match
            #       the order that we initialized ConsistentRandomTransform so
            #       the correct scale is applied to each image
            prod_img_dataset = None
            for product in filtered_products:
                if any(skip_product in product for skip_product in skip_products):
                    continue
                # TODO: should not be necessary if we filter bad files out first
                try:
                    if prod_img_dataset is not None:
                        prod_img_dataset.close()
                    prod_img_dataset = self._load_img(
                        os.path.join(file_path, prod_imgs[product]),
                        product.split("-")[-1] if product not in self.SEPARATE_FOLDER_PRODUCTS else None,
                    )
                except Exception as e:
                    print("ERROR")
                    print(e)
                    print(file_path)
                    self._bad_hits["load_error"] += 1
                    if prod_img_dataset is not None:
                        prod_img_dataset.close()
                    return self.__getitem__(idx + 1)

                if crop_bounds is None:
                    # Move to nearest pixel from largest GSD product
                    # TODO: if we resample the largest product, it might not be the largest product anymore!
                    # TODO: should we sort products largest to smallest by product_changes?
                    largest_prod = prod_img_dataset
                    largest_GSD = self.GSD(filtered_products[0])

                    crop_center_points = uniform(
                        footprints_df.iloc[sample_try]["geometry"], batch_size, rng=np.random.get_state()[1]
                    )
                    if self.batch_size == 1:
                        crop_center_points = np.array([crop_center_points.xy])
                    else:
                        crop_center_points = np.array([p.xy for p in crop_center_points.geoms])  # [B, 2, 1]

                    # Make sure that N and S UTM zones are handled correctly
                    if Proj(footprints_df.crs) != Proj(largest_prod.crs.crs_wkt):
                        transformer = Transformer.from_crs(footprints_df.crs, largest_prod.crs.crs_wkt, always_xy=True)

                        crop_center_points = transformer.transform(
                            crop_center_points[:, 0, 0], crop_center_points[:, 1, 0]
                        )
                        crop_center_points = np.stack(crop_center_points, axis=1).reshape(batch_size, 2, 1)

                    crop_corner_points = crop_center_points - self.ground_cover / 2

                    # pixel align
                    crop_aligned_min_x = (
                        largest_prod.x.sel(x=crop_corner_points[:, 0, 0] + largest_GSD / 2, method="nearest").values
                        - largest_GSD / 2
                    )

                    crop_aligned_min_y = (
                        largest_prod.y.sel(y=crop_corner_points[:, 1, 0] + largest_GSD / 2, method="nearest").values
                        - largest_GSD / 2
                    )

                    crop_aligned_min_xy = np.stack([crop_aligned_min_x, crop_aligned_min_y], axis=1).astype(int)

                    crop_bounds = np.stack(
                        [
                            crop_aligned_min_xy[:, 0],
                            crop_aligned_min_xy[:, 1],
                            crop_aligned_min_xy[:, 0] + self.ground_cover,
                            crop_aligned_min_xy[:, 1] + self.ground_cover,
                        ],
                        axis=1,
                    )

                    crop_crs = prod_img_dataset.rio.crs
                    crop_crs_wkt = prod_img_dataset.crs.crs_wkt

                product_gsd = self.GSD(product)
                img_size = math.ceil(self.ground_cover / product_gsd)

                # Handle N and S UTMs difference
                if prod_img_dataset.crs.crs_wkt != crop_crs_wkt:  # TODO remove or keep?
                    transformer = Transformer.from_crs(crop_crs_wkt, prod_img_dataset.crs.crs_wkt, always_xy=True)

                    # Transform the coordinates of the bounding box
                    # (x_min, y_min, x_max, y_max) = crop_bounds
                    x_min = crop_bounds[:, 0]
                    y_min = crop_bounds[:, 1]
                    x_max = crop_bounds[:, 2]
                    y_max = crop_bounds[:, 3]
                    x_min_transformed, y_min_transformed = transformer.transform(x_min, y_min)
                    x_max_transformed, y_max_transformed = transformer.transform(x_max, y_max)

                    prod_crop_bounds = np.round(
                        np.stack([x_min_transformed, y_min_transformed, x_max_transformed, y_max_transformed], axis=1)
                    ).astype(np.int32)
                else:
                    prod_crop_bounds = crop_bounds

                X = np.stack([
                    np.arange(prod_crop_bounds[i, 0], prod_crop_bounds[i, 2], self.GSD(product))
                    for i in range(batch_size)
                ])
                Y = np.stack([
                    np.arange(prod_crop_bounds[i, 1], prod_crop_bounds[i, 3], self.GSD(product))
                    for i in range(batch_size)
                ])
                X += int(self.GSD(product) / 2)
                Y += int(self.GSD(product) / 2)

                if "S1" in product:
                    if "1SSH" in prod_imgs[product]:
                        single_band_S1 = (f"{product}-1SSH", product)
                    elif "1SSV" in prod_imgs[product]:
                        single_band_S1 = (f"{product}-1SSV", product)

                product_key = product

                if (
                    abs(Y).max() > abs(prod_img_dataset.y).max() or abs(X).max() > abs(prod_img_dataset.x).max()
                ):  # TODO: I think we can remove this now
                    self._bad_hits["out_of_bounds"] += 1
                    found_all_products = False
                    if "S2" in product:
                        skip_products.append(product)
                    if product_key in product_imgs:
                        product_imgs.pop(product_key)
                    break

                for b in range(batch_size):
                    x = X[b]
                    y = Y[b]

                    # NOTE: possibly faster method, works nice with new bounds
                    # #extract the indices of the image
                    prod_img = prod_img_dataset.sel(x=x, y=y, method="nearest")
                    # prod_img = prod_img_dataset.rio.clip_box(*crop_bounds, crs=crop_crs)

                    if prod_img.sizes["x"] != img_size or prod_img.sizes["y"] != img_size:
                        self._bad_hits["size_mismatch"] += 1
                        # breakpoint()
                        found_all_products = False
                        if "S2" in product:
                            skip_products.append(product)
                        if product_key in product_imgs:
                            product_imgs.pop(product_key)
                        break

                    # Convert to numpy array
                    try:
                        prod_img = prod_img.bands.values
                    except Exception as e:
                        print(e)
                        print(f"corrupt file error for prod file path: {os.path.join(file_path, prod_imgs[product])}")
                        self._bad_hits["corrupt_file"] += 1
                        found_all_products = False
                        if "S2" in product:
                            skip_products.append(product)
                        if product_key in product_imgs:
                            product_imgs.pop(product_key)
                        break

                    nan_ratio = np.isnan(prod_img).sum(axis=(1, 2)) / (prod_img.shape[1] * prod_img.shape[2])
                    if (nan_ratio > self.max_nan_ratio).any():
                        self._bad_hits["nan_ratio"] += 1
                        # self.debug_plot_footprints(footprint, file_path, prod_img_paths, crop_center_point, crop_bounds, crs_wkt=crop_crs_wkt, idx=idx, nan_ratio=nan_ratio)
                        found_all_products = False
                        if "S2" in product:
                            skip_products.append(product)
                        if product_key in product_imgs:
                            product_imgs.pop(product_key)
                        break

                    if product_key not in product_imgs:
                        product_imgs[product_key] = []

                    product_imgs[product_key].append(prod_img)

            if found_all_products:  # Found all products for a given sample tile, finish the loop
                # TODO: rewrite this to try N samples number of times and then at the end takes
                # the intersection of the products found (i.e, if we don't find S1 thats ok...)
                break

            # NOTE: this below doubles the speed
            found_important_products = False
            # When we are not strict, we only look for 'important products' and finish the loop
            if not self.strict_data_mode:
                for product in product_imgs.keys():
                    if any(important_product in product for important_product in self.important_products):
                        found_important_products = True
                        break

                if found_important_products:
                    break

        if prod_img_dataset is not None:
            prod_img_dataset.close()

        # Fill missing products with dummy data, will be removed afterward, needed for consistent transforms
        if not found_all_products:  # TODO: double check with S1 modes
            if self.strict_data_mode:
                self._bad_hits["no_sample"] += 1
                return self.__getitem__(idx + 1)

            if not found_important_products:
                self._bad_hits["no_important_products"] += 1
                return self.__getitem__(idx + 1)

        # Load ERA5 data
        if self.era5_data:  # TODO: move this further up if we want strict era5 data mode
            fp = Path(file_path)  # TODO: check tz info
            try:
                file_date = footprints_df.iloc[sample_try]["timestamp"]
            except Exception as e:
                print(f"error for file path: {fp}", e)
                file_date = datetime.strptime(str(fp.relative_to(fp.parents[2])), "%Y/%m/%d")
            try:
                era5_land_data = self.get_era5_product(file_date, crop_center_points, crop_crs_wkt, era5_land=True)
                era5_data = None  # self.get_era5_product(file_date, crop_center_points, crop_crs_wkt, era5_land=False) # TODO add back once we have the data
            except Exception as e:
                print(f"era5 error for file date: {file_date}, path {fp}", e)
                era5_land_data = torch.zeros(
                    batch_size,
                    len(self.era5_land_products),
                    self.era5_land_patch_size,
                    self.era5_land_patch_size,
                    dtype=torch.float32,
                )
                era5_land_data[:] = float("nan")
                era5_land_data = {
                    self.ERA5_LAND_PRODUCT_MAP[p]: era5_land_data[:, i] for i, p in enumerate(self.era5_land_products)
                }
                era5_data = None

        # NOTE: the order that we iterate through products needs to match
        #       the order that we initialized ConsistentRandomTransform so
        #       the correct scale is applied to each image
        # consistent cropping in custom transforms has an internal counter,
        # only trigger counting once we are sure we have all the product images
        # NOTE: I think this new solution does not have this problem
        # NOTE: from here the product will be potentially resized

        # Generating random crop params consistent across all products
        self.cont_cons_rand_crop.generate_crop_params()
        for product, prod_img in product_imgs.items():
            if isinstance(prod_img, list):
                prod_img = np.stack(prod_img, axis=0)
            product_imgs[product] = self.custom_transform[product](prod_img)
        self.cons_horiz_flip.count = 0  # Reset the counter to ensure consistent flip
        self.cons_vertical_flip.count = 0  # Reset the counter to ensure consistent flip

        # TODO: transform / normalize era5 data, now it is done automatically in get_era5_product

        # Ugly hack to handle single band S1 products
        if single_band_S1 and single_band_S1[1] in product_imgs:
            product_imgs[single_band_S1[0]] = product_imgs[single_band_S1[1]]
            del product_imgs[single_band_S1[1]]

        self._hits += 1
        if self._hits % 100 == 0:
            worker_info = get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id

            if worker_id == 0:
                print(f"Worker {worker_id}, idx: {idx}, hits: {self._hits}")
                print(f"crop hit rate: {self._hits / (self._hits + self._bad_hits.total()) * 100:.2f}%")
                print("Bad hits: ")
                self._print_bad_hits()
                # print(f"Cache info load img: {self._load_img.cache_info()}")

        # TODO: rewrite as named dict return
        if self.full_return and self.era5_data:
            return product_imgs, era5_land_data, era5_data, idx
        elif self.era5_data:
            return product_imgs, era5_land_data, era5_data
        elif self.full_return:
            return product_imgs, idx
        return product_imgs
