import os
import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import xarray as xr
from shapely.geometry import Polygon, box
from torch.utils.data import Dataset

from fm4cs.data.fm4cs_dataset_base import FM4CSDatasetBase

# warnings.filterwarnings("error")


def print_node(node, indent: int = 0, node_str=""):
    match node:
        case RootNode(children=children):
            node_str += "RootNode:"
            for child in children:
                node_str += "\n"
                node_str += print_node(child, indent + 1)
        case Node(file=file, children=children, footprint=footprint):
            node_str += f"{'  ' * indent}Node({Path(file).name}, {footprint.area.item()}):"
            for child in children:
                node_str += "\n"
                node_str += print_node(child, indent + 1)
        case Leaf(file=file, footprint=footprint):
            return f"{'  ' * indent}Leaf({Path(file).name}, {footprint.area.item()})"
    return node_str


# TODO: update footprint to be geoseries and remove crs from node and leaf, or actually use polygon as footprint


@dataclass
class RootNode:
    children: list[Union["Node", "Leaf"]]
    level: int = 0
    footprint: Polygon | None = None
    crs: str | None = None
    timestamp: str | None = None

    def __repr__(self):
        return print_node(self)


@dataclass
class Node:
    file: str
    products: list[str]
    children: list[Union["Node", "Leaf"]]
    footprint: Polygon | None = None
    crs: str | None = None
    timestamp: str | None = None
    level: int | None = None
    parent: Union["Node", "RootNode", None] = None

    def __repr__(self):
        return print_node(self)


@dataclass
class Leaf:
    file: str
    products: list[str]
    footprint: Polygon | None = None
    crs: str | None = None
    timestamp: str | None = None
    level: int | None = None
    parent: Union["Node", "RootNode"] | None = None

    def __repr__(self):
        return print_node(self)


SampleNode = RootNode | Node | Leaf


class FootprintBuilder(Dataset, FM4CSDatasetBase):
    def __init__(
        self,
        paths: str | Iterable[str],
        kml_path: str | None = None,
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
        overwrite=False,
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
            legacy_band_names=False,
            **kwargs,
        )

        self.save_fig = save_fig
        self.verbose = False
        self.overwrite = overwrite

        self.file_paths, self.file_path_map = self.load_files(split=split, strict=False)
        print(f"Loaded {len(self.file_paths)} files")

        if out_dir is None:
            out_dir = self.get_metadata_dir()
        self.out_dir = Path(out_dir)

        self._tile_bounds = None
        self.kml_path = kml_path

    @property
    def tile_bounds(self):
        if self._tile_bounds is None:
            if self.kml_path is not None:
                import fiona

                fiona.drvsupport.supported_drivers["kml"] = "rw"
                fiona.drvsupport.supported_drivers["KML"] = "rw"
                kml_gdf = gpd.read_file(self.kml_path, driver="LIBKML")
                self._tile_bounds = kml_gdf
                return self._tile_bounds
            else:
                return None
        else:
            return self._tile_bounds

    def get_tile_footprint(self, tile):
        def extract_epsg(html_description):
            # Define the regex pattern to extract the EPSG value
            pattern = re.compile(r"<b>EPSG</b></font></td><td.*?> <font.*?>(\d+)</font></td>")

            # Search for the pattern in the HTML description
            match = pattern.search(html_description)

            # If a match is found, return the EPSG value, else return None
            if match:
                return match.group(1)
            else:
                return None

        tile_data = self.tile_bounds[self.tile_bounds["Name"] == tile]
        if tile_data.empty:
            print(f"Could not find tile {tile}")
            return None
        desc = tile_data["Description"].item()
        geom = tile_data["geometry"].item()

        dst_epsg = extract_epsg(desc)
        footprint = next(iter(geom.geoms))
        footprint = Polygon([(x, y) for x, y, z in footprint.exterior.coords])  # Drop z

        src_epsg = "4326"  # WGS84

        footprint = gpd.GeoSeries([footprint], crs=f"EPSG:{src_epsg}")
        footprint = footprint.to_crs(f"EPSG:{dst_epsg}")

        footprint = footprint.buffer(
            -(2**0.5) * self.ground_cover / 2, cap_style="square", join_style="bevel", resolution=0
        ).set_crs(f"EPSG:{dst_epsg}")

        return footprint

    def __len__(self):
        return len(self.file_paths)

    def load_img_paths(self, file_path, product):
        """
        Finds all images for a given product and product gsd at a given file path
        """
        product_gsd = product.split("-")[-1]

        # Build list of image paths for the product
        if product in self.file_path_map[file_path]:
            img_paths = [
                os.path.join(file_path, rel_img_path) for rel_img_path in self.file_path_map[file_path][product]
            ]
        else:
            img_paths = []

        # filter files on our product gsd
        netcdf_product_group = product_gsd
        if product in self.SEPARATE_FOLDER_PRODUCTS:
            # These products are stored in separate folders
            netcdf_product_group = None

        return img_paths, netcdf_product_group

    def get_footprint(self, prod_img_meta, crop_crs_wkt):
        footprint_meta = prod_img_meta.source_meta.footprint
        if not isinstance(footprint_meta, list):
            footprint_meta = [footprint_meta]

        new_footprint = gpd.GeoSeries.from_wkt(footprint_meta, crs=prod_img_meta.source_meta.footprint_srs)

        new_footprint = new_footprint.to_crs(crop_crs_wkt)

        if len(new_footprint) > 1:
            # print("multiple footprints, merging") # TODO: not sure if merging works, but anyhow, with buffering I think we are ok
            new_footprint = new_footprint.buffer(10, cap_style="square", join_style="bevel", resolution=0)
            new_footprint = gpd.GeoSeries(new_footprint.union_all()).set_crs(crop_crs_wkt)
            new_footprint = new_footprint.buffer(  # TODO: try without / 2
                -(2**0.5) * self.ground_cover / 2 - 10, cap_style="square", join_style="bevel", resolution=0
            )
        else:  # TODO: remove buffer from here! Do it in the pruning step
            # TODO: find out if buffer is slower or faster than offset_curve and convex_hull
            # new_footprint = new_footprint.offset_curve(-self.ground_cover / 2, quad_segs=0, join_style="bevel").convex_hull
            new_footprint = new_footprint.buffer(  # TODO: try without / 2
                -(2**0.5) * self.ground_cover / 2, cap_style="square", join_style="bevel", resolution=0
            ).set_crs(crop_crs_wkt)

        return new_footprint

    def build_sample_graph(self, products: list[str], file_path: str, node: SampleNode, level=0, crop_crs_wkt=None):
        if level >= len(products):  # Ugly
            return

        product = products[level]

        try:
            # Find all images for the product
            img_paths, netcdf_group = self.load_img_paths(file_path, product)

            if len(img_paths) == 0:
                if self.verbose:
                    print(f"Could not find any images for product {product} at file_path: {file_path}")
                return self.build_sample_graph(products, file_path, node, level + 1, crop_crs_wkt)

            if crop_crs_wkt is None:
                with xr.open_dataset(
                    img_paths[0],
                    group=netcdf_group,
                    # decode_coords="all",
                    engine="h5netcdf",
                    cache=False,
                ) as prod_img_dataset:
                    crop_crs_wkt = prod_img_dataset.crs.crs_wkt

        except Exception as e:
            print(f"Error loading images for product {product} file_path {file_path}: {e}")

            return self.build_sample_graph(products, file_path, node, level + 1, crop_crs_wkt)

        found_at_least_one = False
        for img_path in img_paths:
            try:
                # Get source metadata footprint, NB! the netcdf might contain cropped images, but the footprints are not updated!
                group = netcdf_group if product in self.SEPARATE_FOLDER_PRODUCTS else None
                # TODO just use filename?... _20191020T110754_20191020T110819_
                with xr.open_dataset(img_path, group=group, cache=False, engine="h5netcdf") as prod_img_meta:
                    footprint = self.get_footprint(prod_img_meta, crop_crs_wkt)
                    timestamp = None
                    try:
                        timestamp_strs = prod_img_meta.source_meta.endposition
                        if not isinstance(timestamp_strs, list):
                            timestamp_strs = [timestamp_strs]
                        for timestamp_str in timestamp_strs:
                            if timestamp_str.endswith("Z"):
                                timestamp_str = timestamp_str[:-1]
                            ts = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
                            if timestamp is None or ts > timestamp:
                                timestamp = ts
                    except Exception as e:
                        print(f"Error parsing timestamp for {img_path}: {e}")

                # TODO: only necessary for S3 products (I think, double check with Jarle)
                # Get bounding box of the image
                with xr.open_dataset(img_path, group=netcdf_group, cache=False, engine="h5netcdf") as prod_img:
                    prod_img.rio.write_crs(prod_img.crs.crs_wkt, inplace=True)
                    bounds = prod_img.rio.bounds()
                    bounds = gpd.GeoSeries(box(*bounds), crs=prod_img.crs.crs_wkt)

                # Intersect the footprint with the bounding box
                footprint = self.intersect_footprints(footprint, bounds, crop_crs_wkt, limit_area=False)

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

            if footprint is None:
                wrn_msg = f"Footprint is None for image {img_path}"
                warnings.warn(wrn_msg)
                continue

            found_at_least_one = True
            node_products = [prod for prod in self.products if self.SUBDIR_MAP[prod] == self.SUBDIR_MAP[product]]

            if "S1" in product:
                if "1SDH" in img_path or "1SSH" in img_path:
                    node_products = [prod for prod in node_products if "HH" in prod]
                elif "1SDV" in img_path or "1SSV" in img_path:
                    node_products = [prod for prod in node_products if "VV" in prod]

            if level == len(products) - 1:
                leaf = Leaf(
                    file=str(Path(img_path).relative_to(file_path)),
                    products=node_products,
                    footprint=footprint,
                    crs=crop_crs_wkt,
                    timestamp=timestamp,
                    level=level,
                    parent=node,
                )
                node.children.append(leaf)
            else:
                new_node = Node(
                    file=str(Path(img_path).relative_to(file_path)),
                    products=node_products,
                    children=[],
                    footprint=footprint,
                    crs=crop_crs_wkt,
                    timestamp=timestamp,
                    level=level,
                    parent=node,
                )
                node.children.append(new_node)
                self.build_sample_graph(products, file_path, new_node, level + 1, crop_crs_wkt)

        # Skip product and continue
        if not found_at_least_one:
            return self.build_sample_graph(products, file_path, node, level + 1, crop_crs_wkt)

    def intersect_footprints(self, f1, f2, crs, limit_area=True):
        # footprint = footprint.to_crs(crop_crs_wkt) # gpd.GeoSeries([shapely.geometry.box(*footprint)])
        f1 = f1.to_crs(crs)
        f2 = f2.to_crs(crs)
        if not f1.intersects(f2).values[0]:
            return None
        # TODO: add check to see if footprint is one whole polygon

        new_footprint = shapely.intersection(f1, f2)
        new_footprint = new_footprint.set_crs(crs)

        # TODO: parameterize and explore this
        if limit_area and (
            (new_footprint.area < 2 * self.ground_cover**2).item()
            or (new_footprint.boundary.shortest_line(new_footprint.centroid).length < self.ground_cover / 2).item()
        ):
            return None

        return new_footprint

    def find_all_sample_areas(
        self,
        node: SampleNode,
        sample_areas: list[Leaf],
    ) -> bool:
        match node:
            case RootNode(children=children, footprint=footprint):
                # Iterate over all children
                for child in children:
                    child.timestamp = (
                        max(child.timestamp, node.timestamp) if child.timestamp is not None else node.timestamp
                    )
                    self.find_all_sample_areas(child, sample_areas)

            case Node(children=children, footprint=footprint, parent=parent):
                # If we have a parent node, try to intersect footprint
                if isinstance(parent, Node):
                    if "S1" in node.products[0] and "S1" in parent.products[0]:
                        # Check if we have the same path, if not, we should skip
                        if not Path(node.file).name == Path(parent.file).name:
                            return False

                    new_footprint = self.intersect_footprints(footprint, parent.footprint, parent.crs)
                    # Continue with the new footprint
                    if new_footprint is not None:
                        node.footprint = new_footprint

                    # If we cannot intersect, try other permutations, either (node.parent -> node.children) or (node -> node.children, exluding parent)
                    else:
                        # Create a new node with our parent as node, and our children as children, skipping us, and then continue
                        new_node = Node(
                            file=parent.file,
                            products=parent.products,
                            children=deepcopy(children),
                            footprint=parent.footprint,
                            crs=parent.crs,
                            parent=parent.parent,
                            timestamp=parent.timestamp,
                        )
                        for child in new_node.children:
                            child.parent = new_node

                        success_new_node = self.find_all_sample_areas(new_node, sample_areas)

                        # Create a new node with our children as children, and our parent's parent as parent, skipping our parent and then continue
                        other_new_node = deepcopy(node)
                        other_new_node.parent = parent.parent
                        success_other_new_node = False
                        # If we have a parent, we should intersect with the parent's footprint,
                        if other_new_node.parent.footprint is not None:
                            other_new_footprint = self.intersect_footprints(
                                other_new_node.footprint, other_new_node.parent.footprint, other_new_node.crs
                            )
                            # Only continue if we have a valid intersection
                            if other_new_footprint is not None:
                                other_new_node.footprint = other_new_footprint

                                for child in other_new_node.children:
                                    child.parent = other_new_node

                                # else:
                                #     msg = f"Something is wrong!, our parent (a rootnode) does not have an intersecting footprint with us! \
                                #             \nParent:\n{other_new_node.parent} Node:\n{other_new_node} \n file: {other_new_node.file}"
                                #     raise ValueError(msg)

                                success_other_new_node = self.find_all_sample_areas(other_new_node, sample_areas)

                        # If we have at least one success, we are not a leaf node
                        if not (success_new_node or success_other_new_node):
                            # We are now a leaf node, and we should add our footprint to the sample areas
                            new_leaf = Leaf(
                                file=node.file,
                                products=node.products,
                                footprint=node.footprint,
                                crs=node.crs,
                                timestamp=node.timestamp,
                                parent=node.parent,
                            )
                            sample_areas.append(new_leaf)

                        return True

                elif isinstance(parent, RootNode) and parent.footprint is not None:
                    new_footprint = self.intersect_footprints(footprint, parent.footprint, node.crs)
                    if new_footprint is None:
                        msg = f"Something is wrong!, our parent (a rootnode) does not have an intersecting footprint with us! RootNode footprint, crs: {parent.footprint, parent.crs} Node file, products, footprint, crs: {node.file, node.products, node.footprint, node.crs}"
                        raise ValueError(msg)
                    node.footprint = new_footprint

                node.timestamp = (
                    max(node.timestamp, parent.timestamp) if node.timestamp is not None else parent.timestamp
                )

                # Parent is root node or we have successfully intersected footprints
                successes = []
                for child in children:
                    success = self.find_all_sample_areas(child, sample_areas)
                    successes.append(success)

                # We are now a leaf node, and we should add our footprint to the sample areas
                if not any(successes):
                    new_leaf = Leaf(
                        file=node.file,
                        products=node.products,
                        footprint=node.footprint,
                        crs=node.crs,
                        timestamp=node.timestamp,
                        parent=node.parent,
                    )
                    sample_areas.append(new_leaf)

                return True

            case Leaf(footprint=footprint, parent=parent):
                if isinstance(parent, Node) and "S1" in node.products[0] and "S1" in parent.products[0]:
                    # Check if we have the same path, if not, we should skip
                    if not Path(node.file).name == Path(parent.file).name:
                        return False

                if isinstance(parent, Node) or (isinstance(parent, RootNode) and parent.footprint is not None):
                    new_footprint = self.intersect_footprints(footprint, parent.footprint, parent.crs)
                    if new_footprint is None:
                        return False

                    node.footprint = new_footprint

                node.timestamp = (
                    max(node.timestamp, parent.timestamp) if node.timestamp is not None else parent.timestamp
                )

                sample_areas.append(node)
                return True

    @staticmethod
    def _get_samples(node):
        # Sanity checks
        if not isinstance(node, Leaf):
            warnings.warn(f"Node is not a leaf node, but a {type(node)}")

        _iter_node = node
        _iter_node_parent = _iter_node.parent
        while _iter_node_parent.footprint is not None:
            if (_iter_node.footprint.area.item() - _iter_node_parent.footprint.area.item()) > 100:
                warnings.warn(
                    f"Child footprint is larger than parent footprint, node parent file: {_iter_node_parent.file} \n {_iter_node_parent}"
                )
                break
            # _iter_node = _iter_node_parent
            if isinstance(_iter_node_parent, RootNode) or _iter_node_parent.footprint is None:
                break
            _iter_node_parent = _iter_node_parent.parent

        sample_traj = {
            **{prod: node.file for prod in node.products},
            "footprint": node.footprint[0],
            "area": node.footprint.area.item(),
            "timestamp": node.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f") if node.timestamp is not None else None,
            "crs": node.crs,
        }

        traverse_node = node
        while not isinstance(traverse_node, RootNode):
            for prod in traverse_node.products:
                sample_traj[prod] = traverse_node.file
            traverse_node = traverse_node.parent

        return sample_traj

    # @profile(stream=mem_logs)
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        tile = Path(file_path).parents[2].name[1:]

        out_dir = self.out_dir
        fp = Path(file_path)
        # TODO: check tz info
        file_date = datetime.strptime(str(fp.relative_to(fp.parents[2])), "%Y/%m/%d")

        relative_path = fp.relative_to(fp.parents[3])
        out_dir = out_dir / relative_path

        out_filename = self.get_metadata_cache_name()

        out_file = out_dir / out_filename

        if out_file.exists() and not self.overwrite:
            # TODO: read in gdf?
            return (file_path, None, None, True)

        out_dir.mkdir(parents=True, exist_ok=True)

        # Sorting products after SAMPLE_ORDER
        sample_products = sorted(self.products, key=lambda x: self.SAMPLE_ORDER.index(x.split("-")[0]))

        filtered_sample_products = []
        for sample_product in sample_products:
            product_subdir = self.SUBDIR_MAP[sample_product]
            if not any(product_subdir == self.SUBDIR_MAP[prod] for prod in filtered_sample_products):
                filtered_sample_products.append(sample_product)

        sample_products = filtered_sample_products

        # Build the sample graph
        sample_graph = RootNode(children=[], level=0, timestamp=file_date)

        if self.tile_bounds is not None:
            sample_graph.footprint = self.get_tile_footprint(tile)
            if sample_graph.footprint is not None:
                sample_graph.crs = sample_graph.footprint.crs

        self.build_sample_graph(sample_products, file_path, sample_graph)

        # Finding a suitable bounding box to sample from, from the products, possibly multiple files for each sensor
        sample_graph_traversed = []
        self.find_all_sample_areas(sample_graph, sample_graph_traversed)

        prod_imgs = []
        for sample_traversed in sample_graph_traversed:
            sample = self._get_samples(sample_traversed)
            if sample not in prod_imgs:
                # TODO: Check if we still have duplicates
                prod_imgs.append(sample)

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(prod_imgs)

        if "footprint" not in df.columns:
            # print(f"file path: {file_path}")
            return (file_path, df, 0, False)

        # Convert bounding box to shapely geometry
        if isinstance(df["footprint"][0], (list, tuple)):
            df["geometry"] = df["footprint"].apply(lambda bbox: box(*bbox))
        else:
            df["geometry"] = df["footprint"]

        df = df.drop(columns=["footprint"])

        # ensure all products are in the dataframe
        for product in self.products:
            if product not in df.columns:
                df[product] = None

        # Area times number of products for sampling weighting
        # df["not_null"] = len(self.products) - df[self.products].apply(lambda row: sum(row.isnull()), axis=1)

        # Existing sensors, (not products)
        existing_sensors = df[self.products].T.groupby(df[self.products].columns.str.split("-").str[0]).any().T

        df["area_prod"] = (df["area"] / 100_000**2) * existing_sensors.sum(axis=1)
        # df = df.drop(columns=["not_null"])

        # # Convert the DataFrame to a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=df["crs"][0])

        # TODO: debug and fix this later on
        gdf = gdf.drop(gdf[gdf["area_prod"] == 0].index)

        total_area = gdf["area_prod"].sum()  # TODO: or just area?

        if total_area == 0:
            return (file_path, df, 0, False)

        # Save the GeoDataFrame to the specified file path
        gdf.to_file(out_file, driver="GeoJSON")

        if self.save_fig:
            gdf["id"] = gdf.index
            # plot the bounding boxes
            fig, ax = plt.subplots()
            gdf.plot(
                ax=ax,
                column="id",
                categorical=True,
                alpha=0.15,
                # cmap=plt.cm.tab20,
                edgecolor="black",
                legend=True,
            )

            if sample_graph.footprint is not None:
                gpd.GeoSeries(sample_graph.footprint).plot(
                    ax=ax,
                    alpha=0.1,
                    edgecolor="black",
                    color="red",
                    label="Tile",
                )
            plt.savefig(out_file.with_suffix(".png"))
            plt.close(fig)

        return (file_path, gdf, total_area, True)

    def collate_fn(self, batch):
        return batch
