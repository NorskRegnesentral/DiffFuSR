import random
from functools import reduce


def parse_products(products: list[str] | dict[str, str | None]) -> dict[str, str]:
    if isinstance(products, dict):  # TODO: do iteration here, adding product where change is None
        return products
    p = {}
    for prod in products:
        if ":" in prod:
            product, resample = prod.split(":")
            p[product] = resample
        else:
            p[prod] = None

    return p


# alternative simple solution, just use bounding boxes, precompute all of them


def intersect_2_boxes(box1, box2):
    # box format: [minx, miny, maxx, maxy]

    # For the intersection, the minimums are the maximum values among the minimums,
    # and the maximums are the minimum values among the maximums.
    minx = max(box1[0], box2[0])
    miny = max(box1[1], box2[1])
    maxx = min(box1[2], box2[2])
    maxy = min(box1[3], box2[3])

    # If the boxes do not intersect, return None
    if minx > maxx or miny > maxy:
        return None

    return (minx, miny, maxx, maxy)


def intersect_boxes(boxes):
    return reduce(intersect_2_boxes, boxes)


def get_random_crop(bounds: tuple[int, int, int, int], size: tuple[int, int] | int) -> tuple[int, int, int, int]:
    """
    Returns a random crop box within a bound.
    bounds: (minx, miny, maxx, maxy) bounds of the image in meters
    size: (x, y) size of the crop in meters
    """

    if isinstance(size, (int, float)):
        size = (size, size)

    minx = random.uniform(bounds[0], bounds[2] - size[0])
    miny = random.uniform(bounds[1], bounds[3] - size[1])
    maxx = minx + size[0]
    maxy = miny + size[1]

    return int(minx), int(miny), int(maxx), int(maxy)


def get_random_crop_pixel_aligned(
    shape: tuple[int, int],
    size: tuple[int, int] | int,
    gsd: int,
    min_xy_coords: tuple[int, int] = (0, 0),
    return_indices: bool = False,
) -> tuple[int, int, int, int]:
    """
    Returns a random crop box within a bound that is pixel aligned.
    shape: (x, y) shape of the image in pixels
    size: (x, y) size of the crop in pixels
    gsd: ground sample distance in meters
    min_xy_coords: (x, y) minimum x and y coordinates of the image in meters
    """

    if isinstance(size, int):
        size = (size, size)

    minx = random.randint(0, int(shape[0]) - size[0])
    miny = random.randint(0, int(shape[1]) - size[1])
    maxx = minx + size[0]
    maxy = miny + size[1]

    if return_indices:
        return minx, miny, maxx, maxy

    minx = int(minx * gsd + min_xy_coords[0])
    miny = int(miny * gsd + min_xy_coords[1])
    maxx = int(maxx * gsd + min_xy_coords[0])
    maxy = int(maxy * gsd + min_xy_coords[1])

    return minx, miny, maxx, maxy


def print_raster(raster):
    print(
        f"shape: {raster.rio.shape}\n"
        f"resolution: {raster.rio.resolution()}\n"
        f"bounds: {raster.rio.bounds()}\n"
        f"sum: {raster.sum().item()}\n"
        f"CRS: {raster.rio.crs}\n"
    )
