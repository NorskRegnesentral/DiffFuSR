import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import xarray as xr
from shapely.geometry import box
from torch.utils.data import DataLoader, Dataset, get_worker_info

from fm4cs.data.fm4cs_dataset_base import FM4CSDatasetBase

# warnings.filterwarnings("error")


class FM4CSDatasetIndexBuilder(Dataset, FM4CSDatasetBase):
    def __init__(
        self,
        paths: str | Iterable[str],
        out_dir: str | None = None,
        split: str = "train",
        products: list | dict = [  # noqa: B006
            "S2-10m",
            "S2-20m",
            "S2-60m",
            "S1-10m",
            "S1-60m",
        ],
        img_size: int | None = None,  # Image size in pixels of the smallest GSD product
        ground_cover: float | None = None,
        data_percent: float = 1.0,
        include_filter: str | None = None,
        save_fig=False,
        **kwargs,
    ):
        super().__init__(
            paths=paths,
            split=split,
            products=products,
            img_size=img_size,
            ground_cover=ground_cover,
            data_percent=data_percent,
            include_filter=include_filter,
            **kwargs,
        )

        self.save_fig = save_fig

        self.file_paths, self.file_path_map = self.load_files(split=None, strict=True)  # split)
        print(f"Loaded {len(self.file_paths)} files")

        if out_dir is None:
            out_dir = f"{self.paths}_geometa"
        self.out_dir = Path(out_dir)

    def __len__(self):
        return len(self.file_paths)

    def load_img_paths(self, file_path, product):
        """
        Finds all images for a given product and product gsd at a given file path
        """
        product_name, product_gsd = product.split("-")

        # Build list of image paths for the product
        img_paths = [
            os.path.join(file_path, rel_img_path) for rel_img_path in self.file_path_map[file_path][product_name]
        ]

        # filter files on our product gsd
        netcdf_product_group = product_gsd
        if product_name == "S3":
            # OLCI resampled and SLSTR have different folders
            img_paths = [img_path for img_path in img_paths if self.SUBDIR_MAP[product] in img_path]

        if product in self.SEPARATE_FOLDER_PRODUCTS:
            # These products are stored in separate folders
            netcdf_product_group = None

        return img_paths, netcdf_product_group

    @lru_cache(maxsize=1024)
    def process_product(
        self,
        product: str,
        file_path: str,
        footprint: tuple[float, float, float, float] | None = None,
    ) -> tuple[list[tuple[float, float, float, float]], list[float], list[str]]:
        from fm4cs.data.utils import intersect_2_boxes

        product_name, product_gsd = product.split("-")

        # Find all images for the product
        img_paths, netcdf_group = self.load_img_paths(file_path, product)

        # Load all bounding boxes for the the given bounding box, intersect with the current bounding box if it exists
        bounding_boxes = []
        bounding_box_areas = []
        intersected_img_paths = []
        for img_path in img_paths:
            try:
                group = product_gsd
                if product in self.SEPARATE_FOLDER_PRODUCTS:
                    group = None
                with xr.open_dataset(
                    img_path,
                    group=group,
                    cache=False,
                    engine="h5netcdf",
                ) as prod_img_meta:  # TODO: add some kwargs to only load rio and save time?
                    bounds = prod_img_meta.rio.bounds()
            except Exception as e:
                print(e)
                print(f"Could not load image: {img_path}")
                continue
                # raise e

            if footprint is not None:
                bounds = intersect_2_boxes(footprint, bounds)
                if bounds is None:
                    continue

            (minx, miny, maxx, maxy) = bounds
            area = (maxx - minx) * (maxy - miny)

            if (maxx - minx) < 1.5 * self.ground_cover or (maxy - miny) < 1.5 * self.ground_cover:
                continue

            bounding_boxes.append(bounds)
            bounding_box_areas.append(area)
            intersected_img_paths.append(img_path)

        return bounding_boxes, bounding_box_areas, intersected_img_paths

    def process_product_footprint(
        self,
        product: str,
        file_path: str,
        footprint=None,
        crop_crs_wkt=None,
    ):
        product_name, product_gsd = product.split("-")

        # Find all images for the product # TODO: wrap with try except, product might not exist
        img_paths, netcdf_group = self.load_img_paths(file_path, product)

        if crop_crs_wkt is None:
            with xr.open_dataset(
                img_paths[0],
                group=netcdf_group if product not in self.SEPARATE_FOLDER_PRODUCTS else None,
                # decode_coords="all",
                engine="h5netcdf",
                cache=False,
            ) as prod_img_dataset:
                crop_crs_wkt = prod_img_dataset.crs.crs_wkt

        # Load all bounding boxes for the the given bounding box, intersect with the current bounding box if it exists
        footprints = []
        footprint_areas = []
        intersected_img_paths = []
        for img_path in img_paths:
            group = netcdf_group if product in ["S3-250m", "S1-10m", "S1-50m", "S1-60m"] else None
            try:
                prod_img_meta = xr.open_dataset(
                    img_path,
                    group=group,
                    cache=False,
                    # decode_coords="all",
                    engine="h5netcdf",
                )
            except Exception as e:
                print("ERROR:")  # NOTE: corrupted files will trigger this exception, should redownload these files...
                print(e)
                print("img_path", img_path)
                print("group", group)
                continue
            footprint_meta = prod_img_meta.source_meta.footprint
            if not isinstance(footprint_meta, list):
                footprint_meta = [footprint_meta]

            new_footprint = gpd.GeoSeries.from_wkt(footprint_meta, crs=prod_img_meta.source_meta.footprint_srs)

            new_footprint = new_footprint.to_crs(crop_crs_wkt)

            if len(new_footprint) > 1:
                # print("multiple footprints, merging")
                new_footprint = new_footprint.buffer(10, cap_style="square", join_style="bevel", resolution=0)
                new_footprint = gpd.GeoSeries(new_footprint.union_all()).set_crs(crop_crs_wkt)
                new_footprint = new_footprint.buffer(
                    -(2**0.5) * self.ground_cover / 2 - 10, cap_style="square", join_style="bevel", resolution=0
                )
            else:
                # TODO: find out if buffer is slower or faster than offset_curve and convex_hull
                # new_footprint = new_footprint.offset_curve(-self.ground_cover / 2, quad_segs=0, join_style="bevel").convex_hull
                new_footprint = new_footprint.buffer(
                    -(2**0.5) * self.ground_cover / 2, cap_style="square", join_style="bevel", resolution=0
                ).set_crs(crop_crs_wkt)

            if footprint is not None:
                # footprint = footprint.to_crs(crop_crs_wkt) # gpd.GeoSeries([shapely.geometry.box(*footprint)])
                if not new_footprint.intersects(footprint).values[0]:
                    continue
                # TODO: add check to see if footprint is one whole polygon

                new_footprint = shapely.intersection(footprint, new_footprint)
                new_footprint = new_footprint.set_crs(crop_crs_wkt)

            # TODO: are these too strict?
            if (new_footprint.area < self.ground_cover**2).values[0] or (
                new_footprint.boundary.shortest_line(new_footprint.centroid).length < self.ground_cover / 2
            ).values[0]:
                continue

            footprints.append(new_footprint)
            footprint_areas.append(new_footprint.area.item())
            intersected_img_paths.append(img_path)

        return footprints, footprint_areas, intersected_img_paths, crop_crs_wkt

    def find_all_sample_areas(
        self,
        products: list[str],
        file_path: str,
        prod_img_datasets: list[dict[str, str]],  # = [{}],
        prod_img_dataset_index: int = 0,
        footprint=None,
        crop_crs_wkt: str | None = None,
        metadata: dict[str, str] | None = {},
    ) -> tuple[tuple[int, int, int, int] | None, dict[str, str] | None]:
        """Recursively find all possible bounding boxes to sample from."""
        if len(prod_img_datasets) == 0:
            prod_img_datasets.append({})

        goto_next_product = False

        # Iterate through each product and find the intersection of the product's footprint with the previous footprint
        # Some products have identical footprints, which are skipped (i.e., S2-10m and S2-20m)
        for product in products:
            product_name, product_gsd = product.split("-")

            if len(prod_img_datasets) > 0:
                for key in prod_img_datasets[prod_img_dataset_index].keys():
                    if key.startswith(product_name) and product not in ["S3-250m"]:  # self.SEPARATE_FOLDER_PRODUCTS:
                        # NOTE: as far as I know it is only S3-250m which has a different footprint than S3-500m and S3-1000m
                        prod_img_datasets[prod_img_dataset_index][product] = prod_img_datasets[prod_img_dataset_index][
                            key
                        ]
                        goto_next_product = True
                        break

            if goto_next_product:
                goto_next_product = False
                continue

            # New product, process bounding box
            bounding_boxes, bounding_box_areas, intersected_img_paths, crop_crs_wkt = self.process_product_footprint(
                product, file_path, footprint, crop_crs_wkt
            )
            if "crs" not in metadata:
                metadata["crs"] = crop_crs_wkt

            # print(f"product: {product}")
            # print(f"bounding boxes: {bounding_boxes}")
            # print(f"intersected img paths: {intersected_img_paths}")
            # print(f"bounding box areas: {bounding_box_areas}")
            # print()

            # Recursively find next sensor product bounding box intersection
            start_idx = len(prod_img_datasets)
            for img_id in range(len(intersected_img_paths)):
                new_prod_img_dataset_index = len(prod_img_datasets)
                prod_img_datasets.append(prod_img_datasets[prod_img_dataset_index].copy())

                img_path = intersected_img_paths[img_id]
                new_bounding_box = bounding_boxes[img_id]
                new_bounding_box_aera = bounding_box_areas[img_id]

                # print(f"product: {product}, index: {new_prod_img_dataset_index} img_path {img_id}: {intersected_img_paths[img_id]}")

                if new_bounding_box is not None:
                    prod_img_datasets[new_prod_img_dataset_index]["bounding_box"] = new_bounding_box
                    prod_img_datasets[new_prod_img_dataset_index]["area"] = new_bounding_box_aera
                    prod_img_datasets[new_prod_img_dataset_index][product] = os.path.relpath(img_path, file_path)
                if len(products) > 1:
                    self.find_all_sample_areas(
                        products[products.index(product) + 1 :],
                        file_path,
                        prod_img_datasets,
                        new_prod_img_dataset_index,
                        new_bounding_box,
                        crop_crs_wkt,
                    )

            # print()
            return

    # @profile(stream=mem_logs)
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # print(f"file path: {file_path}")

        # Sorting products after SAMPLE_ORDER
        sample_products = sorted(self.products, key=lambda x: self.SAMPLE_ORDER.index(x.split("-")[0]))
        # Finding a suitable bounding box to sample from, from the products, possibly multiple files for each sensor
        prod_imgs = []

        # print(f"sampling products: {sample_products}")
        metadata = {}
        self.find_all_sample_areas(sample_products, file_path, prod_img_datasets=prod_imgs, metadata=metadata)
        prod_imgs = [
            prod_img for prod_img in prod_imgs if len(prod_img) == len(sample_products) + 2
        ]  # TODO: remove this, should not be necessary

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(prod_imgs)

        if "bounding_box" not in df.columns:
            return (file_path, prod_imgs, 0, False)

        # Convert bounding box to shapely geometry
        if isinstance(df["bounding_box"][0], list | tuple):
            df["geometry"] = df["bounding_box"].apply(lambda bbox: box(*bbox))
        else:
            df["geometry"] = [feature[0] for feature in df["bounding_box"]]

        df = df.drop(columns=["bounding_box"])

        # # Convert the DataFrame to a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=metadata["crs"])

        # Get the total area of the bounding boxes
        total_area = gdf["area"].sum()

        out_dir = self.out_dir
        fp = Path(file_path)
        relative_path = fp.relative_to(fp.parents[3])
        out_dir = out_dir / relative_path
        out_dir.mkdir(parents=True, exist_ok=True)

        out_products = sorted({self.SUBDIR_MAP[prod] for prod in self.products})
        out_filename = f"{'_'.join(out_products)}_{self.ground_cover}.geojson"
        out_filename = out_filename.replace("/", "_")

        out_file = out_dir / out_filename

        # Save the GeoDataFrame to the specified file path
        gdf.to_file(out_file, driver="GeoJSON")

        if self.save_fig:
            gdf["id"] = gdf.index
            # plot the bounding boxes
            fig, ax = plt.subplots()
            gdf.plot(ax=ax, column="id", alpha=0.15, cmap=plt.cm.tab20, edgecolor="black", legend=True)
            plt.savefig(out_file.with_suffix(".png"))

        return (file_path, prod_imgs, total_area, True)

    def collate_fn(self, batch):
        return batch
