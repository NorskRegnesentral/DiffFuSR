import glob
import math
import multiprocessing as mp
import os
import random
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from pyproj import Transformer
from torchvision import transforms
from tqdm import tqdm

from fm4cs.utils.sentinel import (
    CallWrapper,
    ConsistentRadomHorizontalFlip,
    ConsistentRadomVerticalFlip,
    ControlledConsistentRandomCrop,
)

# from memory_profiler import profile
# mem_logs = open('mem_profile.log','a')


def _to_tensor(x):
    return torch.from_numpy(x.astype(np.float32))


class MinMaxNormalizeTransform:
    def __init__(self, min_value, max_value, scale=1.0, clip=True):
        self.min_value = torch.tensor(min_value)
        self.max_value = torch.tensor(max_value)
        self.scale = scale
        self.clip = clip

    def __call__(self, x):
        # shape = x.shape
        if x.ndim == 4:
            v = (x - self.min_value.view(1, -1, 1, 1)) / (
                self.max_value.view(1, -1, 1, 1) - self.min_value.view(1, -1, 1, 1)
            )
        elif x.ndim == 3:
            v = (x - self.min_value.view(-1, 1, 1)) / (self.max_value.view(-1, 1, 1) - self.min_value.view(-1, 1, 1))
        if self.clip:
            v = v.clip(0, 1)
        return v * self.scale


def clamp_3_sigma(tensor, sigma=1.0):
    return torch.clamp(tensor, -3.0 * sigma, 3.0 * sigma)


def clamp_5_sigma(tensor, sigma=1.0):
    return torch.clamp(tensor, -5.0 * sigma, 5.0 * sigma)


class FM4CSDatasetBase:
    # filename_glob = "S[1-3]"
    filename_glob = "T[0-9]*/[0-9]*/[0-9]*/[0-9]*"
    # filename_regex = ".*"  # r'\.nc$'
    # date_format = "%Y-%m-%dT%H:%M:%S.%fZ" # TODO: update to match folder structure or filename

    # Names used within netCDF files
    NETCDF_PRODUCT_BAND_MAP = {  # noqa: RUF012
        # NOTE: VH+VV and HV+HH are never available at the same time
        "S1-IW-VV-10m": {
            0: "sigma0_vh",
            1: "sigma0_vv",
        },
        "S1-IW-HH-10m": {
            0: "sigma0_hh",
            1: "sigma0_hv",
        },
        "S1-IW-VV-60m": {
            0: "sigma0_vh",
            1: "sigma0_vv",
        },
        "S1-IW-HH-60m": {
            0: "sigma0_hh",
            1: "sigma0_hv",
        },
        "S1-EW-VV-10m": {
            0: "sigma0_vh",
            1: "sigma0_vv",
        },
        "S1-EW-HH-10m": {
            0: "sigma0_hh",
            1: "sigma0_hv",
        },
        "S1-EW-VV-60m": {
            0: "sigma0_vh",
            1: "sigma0_vv",
        },
        "S1-EW-HH-60m": {
            0: "sigma0_hh",
            1: "sigma0_hv",
        },
        # TODO: if not legacy, DELETE in init
        ## LEGACY
        "S1-10m": {
            0: "sigma0_vh",
            1: "sigma0_vv",
        },
        "S1-60m": {
            0: "sigma0_vh",
            1: "sigma0_vv",
        },
        # TODO: # Should we rename S1-60m to S1-EW/IW-60m?
        # NOTE: the band names for S1-60m are the same as S1-10m, see PRODUCT_BAND_NAME_MAP for how we differentiate
        "S2-10m": {0: "B02", 1: "B03", 2: "B04", 3: "B08"},
        "S2-20m": {
            0: "B05",
            1: "B06",
            2: "B07",
            3: "B8A",
            4: "B11",
            5: "B12",
        },
        "S2-60m": {
            0: "B01",
            1: "B09",
        },
        "S3-250m": {  # OLCI resampled to 250m
            0: "Oa01_reflectance",
            1: "Oa02_reflectance",
            2: "Oa03_reflectance",
            3: "Oa04_reflectance",
            4: "Oa05_reflectance",
            5: "Oa06_reflectance",
            6: "Oa07_reflectance",
            7: "Oa08_reflectance",
            8: "Oa09_reflectance",
            9: "Oa10_reflectance",
            10: "Oa11_reflectance",
            11: "Oa12_reflectance",
            12: "Oa13_reflectance",
            13: "Oa14_reflectance",
            14: "Oa15_reflectance",
            15: "Oa16_reflectance",
            16: "Oa17_reflectance",
            17: "Oa18_reflectance",
            18: "Oa19_reflectance",
            19: "Oa20_reflectance",
            20: "Oa21_reflectance",
        },
        "S3-500m": {
            0: "S1_reflectance_an",
            1: "S2_reflectance_an",
            2: "S3_reflectance_an",
            3: "S4_reflectance_an",
            4: "S5_reflectance_an",
            5: "S6_reflectance_an",
        },
        "S3-1000m": {
            0: "S7_BT_in",
            1: "S8_BT_in",
            2: "S9_BT_in",
        },
    }

    # Names used within the dataset
    PRODUCT_BAND_NAME_MAP = {  # noqa: RUF012
        # NOTE: VH+VV and HV+HH are never available at the same time
        "S1-IW-VV-10m": {0: "S1:IW-VH_10", 1: "S1:IW-VV_10"},
        "S1-IW-HH-10m": {0: "S1:IW-HH_10", 1: "S1:IW-HV_10"},
        # TODO: Should we use S1-IW: instead? to specify the mode as a separate product
        "S1-IW-VV-60m": {0: "S1:IW-VH_60", 1: "S1:IW-VV_60"},
        "S1-IW-HH-60m": {0: "S1:IW-HH_60", 1: "S1:IW-HV_60"},
        "S1-EW-VV-10m": {0: "S1:EW-VH_10", 1: "S1:EW-VV_10"},
        "S1-EW-HH-10m": {0: "S1:EW-HH_10", 1: "S1:EW-HV_10"},
        "S1-EW-VV-60m": {0: "S1:EW-VH_60", 1: "S1:EW-VV_60"},
        "S1-EW-HH-60m": {0: "S1:EW-HH_60", 1: "S1:EW-HV_60"},
        # NOTE: differentiating between S1 10m and 60m bands by suffix
        # TODO: if not legacy, DELETE in init
        ## LEGACY
        "S1-10m": {0: "S1:VH_10", 1: "S1:VV_10"},
        "S1-60m": {0: "S1:VH_60", 1: "S1:VV_60"},
        "S2-10m": {0: "S2:Red", 1: "S2:Green", 2: "S2:Blue", 3: "S2:NIR"},
        "S2-20m": {
            0: "S2:RE1",
            1: "S2:RE2",
            2: "S2:RE3",
            3: "S2:RE4", # NIR
            4: "S2:SWIR1",
            5: "S2:SWIR2",
        },
        "S2-60m": {
            0: "S2:CoastAerosal",
            1: "S2:WaterVapor",
        },
        "S3-250m": {  # OLCI resampled to 250m
            0: "S3:Oa01_reflectance",
            1: "S3:Oa02_reflectance",
            2: "S3:Oa03_reflectance",
            3: "S3:Oa04_reflectance",
            4: "S3:Oa05_reflectance",
            5: "S3:Oa06_reflectance",
            6: "S3:Oa07_reflectance",
            7: "S3:Oa08_reflectance",
            8: "S3:Oa09_reflectance",
            9: "S3:Oa10_reflectance",
            10: "S3:Oa11_reflectance",
            11: "S3:Oa12_reflectance",
            12: "S3:Oa13_reflectance",
            13: "S3:Oa14_reflectance",
            14: "S3:Oa15_reflectance",
            15: "S3:Oa16_reflectance",
            16: "S3:Oa17_reflectance",
            17: "S3:Oa18_reflectance",
            18: "S3:Oa19_reflectance",
            19: "S3:Oa20_reflectance",
            20: "S3:Oa21_reflectance",
        },
        "S3-500m": {
            0: "S3:S1_reflectance_an",
            1: "S3:S2_reflectance_an",
            2: "S3:S3_reflectance_an",
            3: "S3:S4_reflectance_an",
            4: "S3:S5_reflectance_an",
            5: "S3:S6_reflectance_an",
        },
        "S3-1000m": {
            0: "S3:S7_BT_in",
            1: "S3:S8_BT_in",
            2: "S3:S9_BT_in",
        },
    }

    # NOTE: using dataset band names to be able to differentiate between S1 10m and 60m bands
    ALL_BAND_NAMES = [name for _, bands in PRODUCT_BAND_NAME_MAP.items() for _, name in bands.items()]

    rgb_bands = ("S2:Blue", "S2:Green", "S2:Red")

    SUBDIR_MAP = {  # noqa: RUF012
        "S1-IW-VV-10m": "S1/IW_GRDH_1S/10m",
        "S1-IW-HH-10m": "S1/IW_GRDH_1S/10m",
        "S1-IW-VV-60m": "S1/IW_GRDH_1S/60m",
        "S1-IW-HH-60m": "S1/IW_GRDH_1S/60m",
        "S1-EW-VV-10m": "S1/EW_GRDM_1S/10m",
        "S1-EW-HH-10m": "S1/EW_GRDM_1S/10m",
        "S1-EW-VV-60m": "S1/EW_GRDM_1S/60m",
        "S1-EW-HH-60m": "S1/EW_GRDM_1S/60m",
        ## Legacy
        "S1-10m": "S1/IW_GRDH_1S/10m",  # TODO: this excludes EW!!! Need to update
        "S1-60m": "S1/IW_GRDH_1S/60m",  # TODO: this excludes EW!!! Need to update
        "S2-10m": "S2",  # /S2MSI2A
        "S2-20m": "S2",  # /S2MSI2A
        "S2-60m": "S2",  # /S2MSI2A
        "S3-250m": "S3/OL_1_EFR",
        "S3-500m": "S3/SL_1_RBT",
        "S3-1000m": "S3/SL_1_RBT",
    }

    NORMALIZATION = {  # noqa: RUF012
    "S2-10m": {
        "S2:Red": {"mean": 0.2250, "std": 0.3002},
        "S2:Green": {"mean": 0.2390, "std": 0.2841},
        "S2:Blue": {"mean": 0.2567, "std": 0.2905},
        "S2:NIR": {"mean": 0.3333, "std": 0.2585},
    },
    "S2-20m": {
        "S2:RE1": {"mean": 0.2834, "std": 0.2883},
        "S2:RE2": {"mean": 0.3134, "std": 0.2642},
        "S2:RE3": {"mean": 0.3216, "std": 0.2505},
        "S2:RE4": {"mean": 0.2062, "std": 0.1780},
        "S2:SWIR1": {"mean": 0.1670, "std": 0.1607},
        "S2:SWIR2": {"mean": 0.3267, "std": 0.2384},
    },
    "S2-60m": {
        "S2:CoastAerosal": {"mean": 0.2144, "std": 0.3013},
        "S2:WaterVapor": {"mean": 0.3343, "std": 0.2572},
    },
    #######################
    "S1-IW-VV-60m": {
        "S1:IW-VH_60": {"mean": -20.4993, "std": 13.2729},
        "S1:IW-VV_60": {"mean": -12.7498, "std": 13.7313},
    },
    "S1-IW-VV-10m": {
        "S1:IW-VH_10": {"mean": -20.6435, "std": 12.6220},
        "S1:IW-VV_10": {"mean": -13.0545, "std": 12.7200},
    },
    "S1-IW-HH-60m": {
        "S1:IW-HH_60": {"mean": -12.5603, "std": 18.8446},
        "S1:IW-HV_60": {"mean": -20.7015, "std": 25.2618},
    },
    "S1-IW-HH-10m": {
        "S1:IW-HH_10": {"mean": -14.1063, "std": 17.2011},
        "S1:IW-HV_10": {"mean": -22.6113, "std": 14.5398},
    },
    "S1-EW-VV-60m": {
        "S1:EW-VH_60": {"mean": -23.2023, "std": 18.3313},
        "S1:EW-VV_60": {"mean": -13.1239, "std": 15.1352},
    },
    "S1-EW-VV-10m": {
        "S1:EW-VH_10": {"mean": -23.5719, "std": 10.7765},
        "S1:EW-VV_10": {"mean": -13.9046, "std": 16.4268},
    },
    ##################
    "S1-EW-HH-60m": {
        "S1:EW-HH_60": {"mean": -12.2785, "std": 18.2552},
        "S1:EW-HV_60": {"mean": -21.8713, "std": 26.8276},
    },
    "S1-EW-HH-10m": {
        "S1:EW-HH_10": {"mean": -12.2785, "std": 18.2552},
        "S1:EW-HV_10": {"mean": -21.8713, "std": 26.8276},
    },
    ###############
    "S3-250m": {
        "S3:Oa01_reflectance": {"mean": 0.3688, "std": 0.1138},
        "S3:Oa02_reflectance": {"mean": 0.3568, "std": 0.1174},
        "S3:Oa03_reflectance": {"mean": 0.3314, "std": 0.1259},
        "S3:Oa04_reflectance": {"mean": 0.3018, "std": 0.1346},
        "S3:Oa05_reflectance": {"mean": 0.2881, "std": 0.1345},
        "S3:Oa06_reflectance": {"mean": 0.2645, "std": 0.1307},
        "S3:Oa07_reflectance": {"mean": 0.2576, "std": 0.1366},
        "S3:Oa08_reflectance": {"mean": 0.2747, "std": 0.1500},
        "S3:Oa09_reflectance": {"mean": 0.2775, "std": 0.1524},
        "S3:Oa10_reflectance": {"mean": 0.2793, "std": 0.1535},
        "S3:Oa11_reflectance": {"mean": 0.2981, "std": 0.1489},
        "S3:Oa12_reflectance": {"mean": 0.3515, "std": 0.1529},
        "S3:Oa13_reflectance": {"mean": 0.0910, "std": 0.0457},
        "S3:Oa14_reflectance": {"mean": 0.1695, "std": 0.0791},
        "S3:Oa15_reflectance": {"mean": 0.3149, "std": 0.1371},
        "S3:Oa16_reflectance": {"mean": 0.3542, "std": 0.1522},
        "S3:Oa17_reflectance": {"mean": 0.3660, "std": 0.1553},
        "S3:Oa18_reflectance": {"mean": 0.3636, "std": 0.1545},
        "S3:Oa19_reflectance": {"mean": 0.2773, "std": 0.1233},
        "S3:Oa20_reflectance": {"mean": 0.1366, "std": 0.0768},
        "S3:Oa21_reflectance": {"mean": 0.3593, "std": 0.1494},
    },
    ###########
    "S3-500m": {
        "S3:S1_reflectance_an": {"mean": 0.3376, "std": 0.1604},
        "S3:S2_reflectance_an": {"mean": 0.3370, "std": 0.1718},
        "S3:S3_reflectance_an": {"mean": 0.4052, "std": 0.1745},
        "S3:S4_reflectance_an": {"mean": 0.0112, "std": 0.0281},
        "S3:S5_reflectance_an": {"mean": 0.1829, "std": 0.1009},
        "S3:S6_reflectance_an": {"mean": 0.1360, "std": 0.0807},
    },
    ##########
    "S3-1000m": {
        "S3:S7_BT_in": {"mean": 286.7439, "std": 44.5894},
        "S3:S8_BT_in": {"mean": 279.2700, "std": 79.1081},
        "S3:S9_BT_in": {"mean": 277.9107, "std": 77.4055},
    },
        # "S1-VV-10m": {  # TODO: update with new stats, or remove for next version
        #     "S1:VH_10": {"mean": -20.8777958, "std": 8.5904760},
        #     "S1:VV_10": {"mean": -13.828976, "std": 12.5036888},
        # },
        # "S1-HH-10m": {  # TODO: update with new stats, or remove for next version
        #     "S1:HH_10": {"mean": -13.828976, "std": 12.5036888},  # TODO: update!!!!!!!!!!!!!!!
        #     "S1:HV_10": {"mean": -20.8777958, "std": 8.5904760},  # TODO: update!!!!!!!!!!!!!!!
        # },
        # "S1-VV-60m": {
        #     "S1:VH_60": {"mean": -20.966874739, "std": 8.9050407},
        #     "S1:VV_60": {"mean": -13.95639942, "std": 10.882105},
        # },
        # "S1-HH-60m": {
        #     "S1:HH_60": {"mean": -13.95639942, "std": 10.882105},  # TODO: update!!!!!!!!!!!!!!!
        #     "S1:HV_60": {"mean": -20.966874739, "std": 8.9050407},  # TODO: update!!!!!!!!!!!!!!!
        # },
        ## LEGACY
        # "S1-10m": {  # TODO: update with new stats, or remove for next version
        #     "S1:VH_10": {"mean": -20.8777958, "std": 8.5904760},
        #     "S1:VV_10": {"mean": -13.828976, "std": 12.5036888},
        # },
        # "S1-60m": {
        #     "S1:VH_60": {"mean": -20.966874739, "std": 8.9050407},
        #     "S1:VV_60": {"mean": -13.95639942, "std": 10.882105},
        # },
        # # Legacy
        # "S3-1000m": {
        #     "S3:S7_BT_in": {"mean": 287.11731124504445, "std": 40.04478454589844},
        #     "S3:S8_BT_in": {"mean": 279.22768817906626, "std": 74.8701400756836},
        #     "S3:S9_BT_in": {"mean": 277.84408833936294, "std": 72.90760040283203},
        # },
        # Stats from ERA5-Land 2016-2024 hourly only 12:00
        # TODO: update with new stats
        "ERA5-Land": {
            "u10": {"mean": -0.5054367780685425, "std": 3.650965452194214},
            "v10": {"mean": 0.3601341247558594, "std": 3.548295021057129},
            "e": {"mean": -0.0004905694513581693, "std": 0.0009033309179358184},
            "pev": {"mean": -0.00248015602119267, "std": 0.004047462251037359},
            "sp": {"mean": 88371.625, "std": 12758.8173828125},
            "ssrd": {"mean": 8133447.5, "std": 8174090.0},
            "swvl2": {"mean": 0.26424697041511536, "std": 0.12406586110591888},
            "swvl3": {"mean": 0.2609078586101532, "std": 0.12165113538503647},
            "stl2": {"mean": 270.33258056640625, "std": 27.16716957092285},
            "stl3": {"mean": 270.1729431152344, "std": 26.814329147338867},
            "d2m": {"mean": 262.22564697265625, "std": 24.547607421875},
            "t2m": {"mean": 269.7378234863281, "std": 27.632509231567383},
            "snowc": {"mean": 48.02663803100586, "std": 48.86762619018555},
            "sd": {"mean": 3.394050359725952, "std": 4.7059550285339355},
            "tp": {"mean": 0.0008007294964045286, "std": 0.002853227313607931},
        },
    }

    SCALE = {  # noqa: RUF012
        # Decibels
        # "S1-IW-VV-10m": {
        #     "S1:IW-VH_10": {"min": -30, "max": 5},
        #     "S1:IW-VV_10": {"min": -30, "max": 5},
        # },
        # "S1-IW-HH-10m": {
        #     "S1:IW-HH_10": {"min": -30, "max": 5},
        #     "S1:IW-HV_10": {"min": -30, "max": 5},
        # },
        # "S1-IW-VV-60m": {
        #     "S1:IW-VH_60": {"min": -30, "max": 5},
        #     "S1:IW-VV_60": {"min": -30, "max": 5},
        # },
        # "S1-IW-HH-60m": {
        #     "S1:IW-HH_60": {"min": -30, "max": 5},
        #     "S1:IW-HV_60": {"min": -30, "max": 5},
        # },
        # "S1-EW-VV-10m": {
        #     "S1:EW-VH_10": {"min": -30, "max": 5},
        #     "S1:EW-VV_10": {"min": -30, "max": 5},
        # },
        # "S1-EW-HH-10m": {
        #     "S1:EW-HH_10": {"min": -30, "max": 5},
        #     "S1:EW-HV_10": {"min": -30, "max": 5},
        # },
        # "S1-EW-VV-60m": {
        #     "S1:EW-VH_60": {"min": -30, "max": 5},
        #     "S1:EW-VV_60": {"min": -30, "max": 5},
        # },
        # "S1-EW-HH-60m": {
        #     "S1:EW-HH_60": {"min": -30, "max": 5},
        #     "S1:EW-HV_60": {"min": -30, "max": 5},
        # },
        ## LEGACY
        # "S1-10m": {
        #     "S1:VH_10": {"min": -30, "max": 5},
        #     "S1:VV_10": {"min": -30, "max": 5},
        # },
        # "S1-60m": {
        #     "S1:VH_60": {"min": -30, "max": 5},
        #     "S1:VV_60": {"min": -30, "max": 5},
        # },
        # Kelvin
        # "S3-1000m": {
        #     "S3:S7_BT_in": {
        #         "min": 213.15,  # - 60 deg C
        #         "max": 333.15,  # + 60 deg C
        #     },
        #     "S3:S8_BT_in": {
        #         "min": 213.15,  # - 60 deg C
        #         "max": 333.15,  # + 60 deg C
        #     },
        #     "S3:S9_BT_in": {
        #         "min": 213.15,  # - 60 deg C
        #         "max": 333.15,  # + 60 deg C
        #     },
        # },
    }

    S1_MODES = ("IW", "EW")
    S1_POLARIZATIONS = ("VV", "HH")

    SAMPLE_ORDER = ("S1", "S2", "S3")

    SEPARATE_FOLDER_PRODUCTS = (
        "S3-250m",
        "S1-10m",
        "S1-IW-VV-10m",
        "S1-IW-HH-10m",
        "S1-EW-VV-10m",
        "S1-EW-HH-10m",
        "S1-60m",
        "S1-IW-VV-60m",
        "S1-IW-HH-60m",
        "S1-EW-VV-60m",
        "S1-EW-HH-60m",
    )

    ERA5_GSD = 31_000  # 31 km
    ERA5_LAND_GSD = 9_000  # 9 km

    ERA5_LAND_PRODUCT_MAP = {  # TODO: update
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "total_evaporation": "e",
        "potential_evaporation": "pev",
        "surface_pressure": "sp",
        "surface_solar_radiation_downwards": "ssrd",
        "volumetric_soil_water_layer_1": "swvl1",
        "volumetric_soil_water_layer_2": "swvl2",
        "volumetric_soil_water_layer_3": "swvl3",
        "volumetric_soil_water_layer_4": "swvl4",
        "soil_temperature_level_1": "stl1",
        "soil_temperature_level_2": "stl2",
        "soil_temperature_level_3": "stl3",
        "soil_temperature_level_4": "stl4",
        "2m_dewpoint_temperature": "d2m",
        "2m_temperature": "t2m",
        "snow_cover": "snowc",
        "snow_depth_water_equivalent": "sd",
        "total_precipitation": "tp",
    }

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
        img_size: int | None = None,  # Image size in pixels of the smallest GSD product
        ground_cover: float | None = None,
        minmax_normalize: bool = False,
        standardize: bool = False,
        resize: bool = False,
        full_return: bool = False,
        data_percent: float = 1.0,
        max_nan_ratio: float = 0.05,
        include_filter: str | None = None,
        exclude_filter: str | None = None,
        era5_data: bool = False,
        era5_data_dir: str | None = None,
        era5_land_products: list[str] | None = None,
        era5_products: list[str] | None = None,
        important_products: list[str] | None = None,
        legacy_nan_handling: bool = False,
        legacy_band_names: bool = False,
        random_seed: int = 42,
        **kwargs,
    ) -> None:
        if isinstance(products, dict):
            product_changes = products
            products = list(product_changes.keys())
            for product, change in product_changes.items():
                if product_changes[product] is None:
                    product_changes[product] = product
            self.product_changes = product_changes
        else:
            self.product_changes = {product: product for product in products}

        if not legacy_band_names:
            for product in products:
                if "S1" in product and (
                    not any(polarization in product for polarization in self.S1_POLARIZATIONS)
                    or not any(mode in product for mode in self.S1_MODES)
                ):
                    msg = f"S1 product {product} will be expanded to include all polarizations, set legacy_band_names=True to use old band names"
                    warnings.warn(msg, UserWarning)

            self.product_changes = dict(
                zip(
                    self.expand_products(self.product_changes.keys()),
                    self.expand_products(self.product_changes.values()),
                    strict=False,
                )
            )

            products = self.expand_products(products)

        else:
            self.product_changes = dict(
                zip(
                    self.minimize_products(self.product_changes.keys()),
                    self.minimize_products(self.product_changes.values()),
                    strict=False,
                )
            )

            products = self.minimize_products(products)

        # if not legacy_band_names:
        #     del self.NETCDF_PRODUCT_BAND_MAP["S1-10m"]
        #     del self.NETCDF_PRODUCT_BAND_MAP["S1-60m"]
        #     del self.PRODUCT_BAND_NAME_MAP["S1-10m"]
        #     del self.PRODUCT_BAND_NAME_MAP["S1-60m"]
        # else:
        #     for exp_product in self.expand_products(["S1-10m", "S1-60m"]):
        #         del self.NETCDF_PRODUCT_BAND_MAP[exp_product]
        #         del self.PRODUCT_BAND_NAME_MAP[exp_product]

        # TODO: need to debug this
        # if legacy_band_names:
        #    del self.SCALE["S3-1000m"]
        # else:
        #    del self.NORMALIZATION["S3-1000m"]

        if important_products is not None:
            self.important_products = important_products
        elif not any("S2" in product for product in products):
            # Typical low res mode
            self.important_products = ("S1", "S3")
        elif not any("S1" in product for product in products):
            # Assuming high res mode
            self.important_products = ("S2",)
        elif not any("S2" in product for product in products) and not any("S1" in product for product in products):
            # Only one left :)
            self.important_products = ("S3",)
        else:
            # Using all, assuming high res mode
            self.important_products = ("S1", "S2")

        print(f"products: {products}")
        print(f"product changes: {self.product_changes}")
        print(f"important products: {self.important_products}")

        assert all(product in self.NETCDF_PRODUCT_BAND_MAP for product in products), (
            f"Invalid product in {products}, available products: {self.NETCDF_PRODUCT_BAND_MAP.keys()}"
        )
        assert all(band in self.ALL_BAND_NAMES for band in discard_bands), (
            f"Invalid band in {discard_bands}, available bands: {self.ALL_BAND_NAMES}"
        )

        assert not (img_size is None and ground_cover is None), "Either img_size or ground_cover must be specified"

        assert not (era5_data and era5_data_dir is None), "era5_data_dir must be specified if era5_data is True"
        assert not (era5_data and era5_land_products is None and era5_products is None), (
            "era5_products and/or era5_land_products must be specified if era5_data is True"
        )

        largest_to_smallest_gsd_products = sorted(products, key=lambda x: self.GSD(x), reverse=True)
        print(f"largest to smallest gsd products: {largest_to_smallest_gsd_products}")
        self.paths = paths
        self.products = largest_to_smallest_gsd_products
        self.split = split
        self.img_size = img_size
        self.ground_cover = ground_cover
        self.minmax_normalize = minmax_normalize
        self.standardize = standardize
        self.resize = resize
        self.full_return = full_return
        self.data_percent = data_percent
        self.discard_bands = discard_bands  # TODO: remove from base?
        self.max_nan_ratio = max_nan_ratio
        self.legacy_nan_handling = legacy_nan_handling
        self.legacy_band_names = legacy_band_names
        self.era5_data_dir = era5_data_dir
        self.era5_data = era5_data
        self.era5_land_products = era5_land_products
        self.era5_products = era5_products

        self.random_seed = random_seed

        self.era5_land_patch_size = math.ceil(self.ground_cover / self.ERA5_LAND_GSD)
        self.era5_patch_size = math.ceil(self.ground_cover / self.ERA5_GSD)

        self.sensors = {p.split("-")[0] for p in self.products}

        print("Split:", self.split)

        self.include_tiles = None
        if include_filter is not None:
            with open(self.include_filter) as f:
                self.incude_tiles = f.readlines()
            self.include_tiles = [
                f.strip().removesuffix("/**").removesuffix("/*").removesuffix("/")
                for f in self.include_tiles
                if "#" not in f
            ]
            print("include_tiles:", self.include_tiles)

        self.exclude_tiles = None
        if exclude_filter is not None:
            with open(exclude_filter) as f:
                self.exclude_tiles = f.readlines()
            self.exclude_tiles = [
                f.strip().removesuffix("/**").removesuffix("/*").removesuffix("/")
                for f in self.exclude_tiles
                if "#" not in f
            ]
            print("exclude_tiles:", self.exclude_tiles)

        self.largest_gsd_product_shape = None

        if self.img_size is None:  # TODO: this is incorrect if we resample the largest pixel product
            self.img_size = int(self.ground_cover / self.GSD(largest_to_smallest_gsd_products[-1]))

        elif self.ground_cover is None:
            self.ground_cover = self.img_size * self.GSD(largest_to_smallest_gsd_products[-1])

        print(f"Ground cover: {self.ground_cover} m")
        print(f"Largest image size: {self.img_size} pixels")

    def get_metadata_dir(self) -> Path:
        out_products = sorted({self.SUBDIR_MAP[prod] for prod in self.products})
        out_footprint_key = f"{Path(self.paths).name}_geometa_{'_'.join(out_products)}_{self.ground_cover}"
        out_footprint_key = out_footprint_key.replace("/", "_")
        out_dir = Path(self.paths).parent / out_footprint_key

        return out_dir

    def get_metadata_cache_name(self) -> str:
        out_products = sorted({self.SUBDIR_MAP[prod] for prod in self.products})
        out_filename = f"{'_'.join(out_products)}_{self.ground_cover}.geojson"
        out_filename = out_filename.replace("/", "_")
        return out_filename

    def expand_products(self, products):
        """
        Expand products to include all modes and polarizations if not already included.
        """
        # TODO: not sure if we should expand the S1_modes, maybe just the polarizations
        expanded_products = []
        for product in products:
            if "S1" in product and (
                not any(polarization in product for polarization in self.S1_POLARIZATIONS)
                or not any(mode in product for mode in self.S1_MODES)
            ):
                mode, polarization = None, None
                p = product.split("-")
                if len(p) == 2:
                    product_name, product_gsd = (*p,)
                elif len(p) == 3:
                    product_name, unknown, product_gsd = (*p,)
                    if unknown in self.S1_MODES:
                        mode = unknown
                    elif unknown in self.S1_POLARIZATIONS:
                        polarization = unknown
                    else:
                        raise ValueError(f"Unknown part in product name: {unknown}")
                else:
                    product_name, mode, polarization, product_gsd = (*p,)
                    print(
                        f"product_name: {product_name}, mode: {mode}, polarization: {polarization}, product_gsd: {product_gsd}"
                    )

                if mode is not None:
                    modes = [mode]
                else:
                    modes = self.S1_MODES

                if polarization is not None:
                    polarizations = [polarization]
                else:
                    polarizations = self.S1_POLARIZATIONS

                for mode in modes:
                    for polarization in polarizations:
                        if len(p) == 2:
                            # Legacy
                            if mode == self.S1_MODES[-1] and product_gsd in ["10m", "60m"]:
                                print(
                                    f"skipping product: [{product_name}-{mode}-{polarization}-{product_gsd}] during expansion"
                                )
                                continue

                        product_key = f"{product_name}-{mode}-{polarization}-{product_gsd}"

                        expanded_products.append(product_key)
            else:
                expanded_products.append(product)
        return expanded_products

    def minimize_products(self, products):
        """
        Minimize products to remove duplicate products with different polarizations.
        """
        # TODO: not sure if we should minimize the S1_modes, maybe just the polarizations
        minimized_products = []
        for product in products:
            if (
                "S1" in product
                and any(polarization in product for polarization in self.S1_POLARIZATIONS)
                and any(mode in product for mode in self.S1_MODES)
            ):
                product_name, _mode, _polarization, product_gsd = product.split("-")
                product_key = f"{product_name}-{product_gsd}"
                if product_key not in minimized_products:
                    minimized_products.append(product_key)
            else:
                minimized_products.append(product)
        return minimized_products

    def lookup_product(self, product):
        """
        Lookup product with different polarizations.
        """
        # TODO: same for lookup_product, maybe just the polarizations
        if (
            "S1" in product
            and any(polarization in product for polarization in self.S1_POLARIZATIONS)
            and any(mode in product for mode in self.S1_MODES)
        ):
            product_name, _mode, _polarization, product_gsd = product.split("-")
            product_key = f"{product_name}-{product_gsd}"
            return product_key
        else:
            return product

    def _get_product_bands(self, product, netcdf=False):
        if netcdf:
            return list(self.NETCDF_PRODUCT_BAND_MAP[product].values())
        return list(self.PRODUCT_BAND_NAME_MAP[product].values())

    def GSD(self, product):
        return int(product.split("-")[-1].replace("m", ""))

    def GSDS(self, resampled=True):
        if resampled:
            return {self.product_changes[product]: self.GSD(self.product_changes[product]) for product in self.products}
        return {product: self.GSD(product) for product in self.products}

    @property
    def available_products(self):
        return [self.product_changes[product] for product in self.products]

    @staticmethod
    def process_tile_dir(
        path_dir: Path,
        dataset_path: Path,
        subdir_to_products_map: dict,
        important_products: tuple,
        strict: bool,
        include_tiles: list | None,
        exclude_tiles: list | None,
        split_tiles: list | None,
        metadata_dir: Path | None,
        metadata_cache_name: str | None,
    ):
        path_dir = Path(path_dir)
        missing_products = Counter()

        # Filter on split
        if split_tiles is not None:
            if path_dir.parents[2].name not in split_tiles:
                # print(f"skipping path: {path_dir} because it is not in split_tiles")
                return

        # Filter on all products existing
        if strict:
            if not all((path_dir / product_subdir).exists() for product_subdir in subdir_to_products_map.keys()):
                # print(f"skipping path: {path_dir} because not all products exist")
                return

        # Filter on metadata
        if metadata_dir is not None:
            if not (metadata_dir / path_dir.relative_to(dataset_path) / metadata_cache_name).exists():
                # print(f"skipping path: {path_dir} because metadata does not exist")
                return

        # match = re.match(filename_regex, os.path.basename(file_dir))
        # if match is not None:

        if include_tiles is not None:
            if not ("/".join(path_dir.parts[-4:]) in include_tiles or path_dir.parents[2].name in include_tiles):
                return
        if exclude_tiles is not None:
            if "/".join(path_dir.parts[-4:]) in exclude_tiles or path_dir.parents[2].name in exclude_tiles:
                return

        product_files_map = {}

        found_all_products = True
        found_S1_IW = True
        found_S1_EW = True
        found_important_products = False
        for product_subdir, products in subdir_to_products_map.items():
            product_files = [str(p.relative_to(path_dir)) for p in (path_dir / product_subdir).glob("**/*.nc")]
            for product in products:
                if len(product_files) > 0:
                    product_files_map[product] = product_files
                    if any(important_product in product for important_product in important_products) and not strict:
                        found_important_products = True
                else:
                    missing_products[product] += 1

                    if "S1" in product:
                        if "IW" in product:
                            found_S1_IW = False
                        elif "EW" in product:
                            found_S1_EW = False
                    else:
                        found_all_products = False
                        if strict:
                            print(f"skipping path: {path_dir} because product {product} does not exist")
                            # TODO: might need an additional flag here to differentiate between footprint builder and dataset
                            break

        if not found_S1_IW and not found_S1_EW:
            found_all_products = False

        if found_all_products or found_important_products:
            return path_dir, product_files_map, missing_products

    def load_files(self, split=None, strict=True) -> tuple[list[str], dict[str, dict[str, list[str]]]]:
        """A list of all files in the dataset.

        Returns:
            All files in the dataset.

        """

        if isinstance(self.paths, str):
            paths: Iterable[str] = [self.paths]
        else:
            paths = self.paths

        subdir_to_products_map = {}
        for product, subdir in self.SUBDIR_MAP.items():
            if product not in self.products:
                continue
            if subdir not in subdir_to_products_map:
                subdir_to_products_map[subdir] = []
            subdir_to_products_map[subdir].append(product)

        # filename_regex = re.compile(self.filename_regex, re.VERBOSE)

        file_dirs = []
        file_path_map = {}
        missing_products = Counter()

        for path in paths:
            if os.path.isdir(path):
                split_tiles = None
                if split is not None:
                    split_file = f"{Path(path)}_{split}_tiles.txt"
                    if os.path.exists(split_file):
                        with open(split_file) as f:
                            split_tiles = f.read().splitlines()
                    else:
                        warnings.warn(
                            f"Could not find split file '{split_file}'. Ignoring split.",
                            UserWarning,
                        )

                process_func = partial(
                    self.process_tile_dir,
                    dataset_path=Path(path),
                    subdir_to_products_map=subdir_to_products_map,
                    important_products=self.important_products,
                    strict=strict,
                    include_tiles=self.include_tiles,
                    exclude_tiles=self.exclude_tiles,
                    split_tiles=split_tiles,
                    metadata_dir=None if not hasattr(self, "metadata_dir") else self.metadata_dir,
                    metadata_cache_name=None if not hasattr(self, "metadata_cache_name") else self.metadata_cache_name,
                )

                pathname = os.path.join(path, "**", self.filename_glob)
                # for r in tqdm(map(process_func, glob.iglob(pathname, recursive=True))):
                with mp.Pool(4) as pool:
                    for r in tqdm(
                        pool.imap_unordered(process_func, glob.iglob(pathname, recursive=True), chunksize=2),
                        desc="Loading files",
                        disable=(hasattr(self, "global_rank") and self.global_rank != 0),
                    ):
                        if r is not None:
                            file_dirs.append(r[0])
                            file_path_map[r[0]] = r[1]
                            missing_products += r[2]

                # TODO: write this in a more meaningful way
                print(f"missing product stats: {missing_products}")

            else:
                warnings.warn(
                    f"Could not find any relevant files for provided path '{path}'. Path was ignored.",
                    UserWarning,
                )

        file_dirs.sort()
        np.random.default_rng(self.random_seed).shuffle(file_dirs)
        file_dirs = file_dirs[: int(self.data_percent * len(file_dirs))]
        file_dirs.sort()
        file_path_map = {p: file_path_map[p] for p in file_dirs}

        return file_dirs, file_path_map

    def _build_transforms(self, is_train, products, product_changes):
        custom_transforms = {}

        # Assume the order is sorted from smallest to largest img_size b/c
        # we sorted in class init
        if not self.resize:
            img_sizes = [self.img_size for _ in products]
        else:
            img_sizes = [math.ceil(self.ground_cover / self.GSD(product_changes[product])) for product in products]
        print(img_sizes)

        # NOTE: this is troublesome
        # cons_rand_crop = ConsistentRandomCrop(
        #     img_sizes, pad_if_needed=True, padding_mode="constant", fill=0
        # )
        cont_cons_rand_crop = ControlledConsistentRandomCrop(
            img_sizes, pad_if_needed=True, padding_mode="constant", fill=0
        )
        self.cont_cons_rand_crop = cont_cons_rand_crop
        cons_horiz_flip = ConsistentRadomHorizontalFlip(len(img_sizes))
        self.cons_horiz_flip = cons_horiz_flip
        cons_vertical_flip = ConsistentRadomVerticalFlip(len(img_sizes))
        self.cons_vertical_flip = cons_vertical_flip
        for idx, product in enumerate(products):
            t = []

            t.append(_to_tensor)

            if self.minmax_normalize:
                if product in self.SCALE:
                    min_value = [
                        self.SCALE[product][band]["min"] for band in self.PRODUCT_BAND_NAME_MAP[product].values()
                    ]
                    max_value = [
                        self.SCALE[product][band]["max"] for band in self.PRODUCT_BAND_NAME_MAP[product].values()
                    ]

                    t.append(MinMaxNormalizeTransform(min_value, max_value, scale=1.0, clip=True))

                else:
                    warnings.warn(
                        f"No min max values found for product {product}, skipping normalization",
                        UserWarning,
                    )

            if self.standardize:
                if product in self.NORMALIZATION:
                    mean = [
                        self.NORMALIZATION[product][band]["mean"]
                        for band in self.PRODUCT_BAND_NAME_MAP[product].values()
                    ]
                    std = [
                        self.NORMALIZATION[product][band]["std"]
                        for band in self.PRODUCT_BAND_NAME_MAP[product].values()
                    ]
                else:
                    mean = [0.0 for _ in self.PRODUCT_BAND_NAME_MAP[product]]
                    std = [1.0 for _ in self.PRODUCT_BAND_NAME_MAP[product]]
                    warnings.warn(
                        f"No normalization values found for product {product}, using mean=0.0, std=1.0",
                        UserWarning,
                    )
                t.append(transforms.Normalize(mean, std))
                t.append(clamp_5_sigma)

            # TODO: can this be before cons_rand_crop?
            if product_changes[product] != product:
                print(f"resampling product: {product}, to: {product_changes[product]}")
                # calculate resampled img size
                (product_change_name, *_) = product_changes[product].split("-")
                product_change_gsd = self.GSD(product_changes[product])
                img_size = math.ceil(self.ground_cover / product_change_gsd)
                print(f"image size: {img_size}")
                t.append(transforms.Resize((img_size, img_size)))

            if is_train:
                call_wrapper = CallWrapper(cont_cons_rand_crop.forward, idx)
                # TODO: check this, when we don't resize images this seems to be wrong
                t.append(call_wrapper)
                t.append(cons_horiz_flip)
                t.append(cons_vertical_flip)
            else:
                # TODO: check this
                # For each grouped product, we will need a different input size
                t.append(transforms.CenterCrop(img_sizes[idx]))

            custom_transforms[product] = transforms.Compose(t)
        return custom_transforms

    # TODO: we need to extract more than one point for low res products, maybe 4x4 pixels

    # stacked products
    def get_era5_product(
        self,
        date: datetime,
        crop_center_points: np.ndarray,
        crop_crs_wkt: str,
        products: list[str] | tuple[str] | None = None,
        interpolate: bool = False,
        era5_land: bool = False,
    ) -> dict[str, np.ndarray] | None:
        batch_size = len(crop_center_points)

        if era5_land:
            era5_data_path = Path(self.era5_data_dir) / f"era5-land_total_{date.year}-{date.month:02d}_hourly.nc"
            era5_patch_size = self.era5_land_patch_size
            era5_normalization = self.NORMALIZATION["ERA5-Land"]
            era5_products = self.era5_land_products
            era5_product_map = self.ERA5_LAND_PRODUCT_MAP
            era5_gsd = self.ERA5_LAND_GSD
        else:
            era5_data_path = Path(self.era5_data_dir) / f"era5_total_{date.year}-{date.month:02d}_hourly.nc"
            era5_patch_size = self.era5_patch_size
            era5_normalization = self.NORMALIZATION["ERA5"]
            era5_products = self.era5_products
            era5_product_map = self.ERA5_PRODUCT_MAP
            era5_gsd = self.ERA5_GSD

        if era5_patch_size == 1:
            era5_points_lonlat = Transformer.from_crs(crop_crs_wkt, "epsg:4326", always_xy=True).transform(
                crop_center_points[:, 0, 0], crop_center_points[:, 1, 0]
            )

        else:
            i = np.arange(-math.floor(era5_patch_size / 2), math.ceil(era5_patch_size / 2), dtype=np.float32)
            j = np.arange(-math.floor(era5_patch_size / 2), math.ceil(era5_patch_size / 2), dtype=np.float32)
            if era5_patch_size % 2 == 0:
                i += 0.5
                j += 0.5
            i *= era5_gsd
            j *= era5_gsd

            crop_x_points = crop_center_points[:, 0, 0, None] + i[None]
            crop_y_points = crop_center_points[:, 1, 0, None] + j[None]

            crop_x_points = crop_x_points.flatten()
            crop_y_points = crop_y_points.flatten()

            era5_points_lonlat = Transformer.from_crs(crop_crs_wkt, "epsg:4326", always_xy=True).transform(
                crop_x_points, crop_y_points
            )

            era5_points_lonlat = (
                era5_points_lonlat[0].reshape(batch_size, era5_patch_size),
                era5_points_lonlat[1].reshape(batch_size, era5_patch_size),
            )

        lon, lat = era5_points_lonlat

        # Convert lon from -180,180 to 0-360
        lon = (lon + 180) % 360

        # era5_data = self.get_mfdataset(era5_data_paths)
        # TODO: try with netcdf4
        with xr.open_dataset(era5_data_path, engine="h5netcdf", lock=False, cache=False, decode_cf=False) as era5_data:
            # era5_data = xr.decode_cf(era5_data)
            # TODO: check date!!! Especially for for new date after 2024 month october
            era5_data_f = era5_data.total.sel(
                valid_time=int(date.timestamp()), method="nearest"
            )  # TODO: do smarter date selection here

            # TODO: check lat/lon shape instead
            if era5_patch_size == 1:
                if interpolate:
                    era5_vars = era5_data_f.interp(latitude=lat, longitude=lon)
                else:
                    era5_vars = era5_data_f.sel(
                        latitude=xr.DataArray(lat, dims="points"),
                        longitude=xr.DataArray(lon, dims="points"),
                        method="nearest",
                    )

                if products is None:
                    products = era5_vars.coords["variable"].values.tolist()
                    product_indexes = range(len(products))
                else:
                    product_indexes = [era5_data_f.variable.values.tolist().index(product) for product in products]

                # TODO: copy product by product or all at once?
                era5_vars = {
                    product: (
                        np.expand_dims(
                            era5_vars[i].values,
                            (
                                1,
                                2,
                            ),
                        )
                        - era5_normalization[product]["mean"]
                    )
                    / era5_normalization[product]["std"]
                    for i, product in zip(product_indexes, products, strict=False)
                }

            else:
                era5_vars_total = [
                    era5_data_f.sel(latitude=lat[b], longitude=lon[b], method="nearest").drop_vars([
                        "latitude",
                        "longitude",
                    ])
                    for b in range(lat.shape[0])
                ]
                era5_vars_total = xr.concat(era5_vars_total, dim="points", coords="minimal", compat="override")

                if products is None:
                    products = era5_vars_total.coords["variable"].values.tolist()
                    product_indexes = range(len(products))
                else:
                    product_indexes = [era5_data_f.variable.values.tolist().index(product) for product in products]
                era5_vars_total = era5_vars_total.values

                # TODO: copy product by product or all at once?
                era5_vars = {
                    product: (_to_tensor(era5_vars_total[:, i]) - era5_normalization[product]["mean"])
                    / era5_normalization[product]["std"]
                    for i, product in zip(product_indexes, products, strict=False)
                }

            era5_vars = {
                era5_product_map[product_name]: era5_vars[era5_product_map[product_name]]
                if era5_product_map[product_name] in era5_vars
                else torch.zeros((lat.shape[0], era5_patch_size, era5_patch_size))
                for product_name in era5_products
            }

            return era5_vars

    def collate_fn(self, batch):
        if self.full_return and self.era5_data:
            product_imgs, era5_land_data, era5_data, paths = zip(*batch, strict=False)
        elif self.era5_data:
            all_product_imgs, era5_land_data, era5_data = batch
        elif self.full_return:
            all_product_imgs, paths = batch
        else:
            all_product_imgs = batch

        collated_imgs = {}
        # for product in self.products:
        #    if product not in all_product_imgs:  # TODO: add flag on or off strict mode
        #        continue
        for product in all_product_imgs:
            product_imgs = all_product_imgs[product]
            single_band_S1 = None
            if "1SSH" in product:
                product = product.removesuffix("-1SSH")
                single_band_S1 = "sigma0_hh"
            elif "1SSV" in product:
                product = product.removesuffix("-1SSV")
                single_band_S1 = "sigma0_vv"

            # Need product name to differentiate between S1 multilooked to different resolutions
            for band_name, prod_band_name in zip(
                self.NETCDF_PRODUCT_BAND_MAP[product].values(),
                self.PRODUCT_BAND_NAME_MAP[product].values(),
                strict=False,
            ):
                prod_bands = self._get_product_bands(product, netcdf=True)

                if single_band_S1:
                    prod_bands = [single_band_S1]
                    if single_band_S1 != band_name:
                        continue

                band_index = prod_bands.index(band_name)
                band_imgs = product_imgs[:, band_index]

                if band_name not in self.discard_bands:
                    if product != self.product_changes[product] and "S1" in product:
                        product_gsd = product.split("-")[-1]
                        product_gsd = product_gsd.replace("m", "")
                        # S1 product has the GSD in the band name, need to update if we resample
                        prod_band_name = prod_band_name.replace(
                            product_gsd, str(self.GSD(self.product_changes[product]))
                        )

                    if (nan_index := torch.isnan(band_imgs)).any():
                        if self.legacy_nan_handling:
                            band_imgs[nan_index] = -0.1
                        else:
                            # TODO: median or mean?
                            band_means = torch.nanmean(band_imgs, dim=(1, 2))[:, None, None].repeat(
                                1, band_imgs.shape[1], band_imgs.shape[2]
                            )
                            band_imgs[nan_index] = band_means[nan_index]

                    collated_imgs[prod_band_name] = band_imgs[:, None, ...]

        # if self.era5_data and era5_land_data is not None:
        #     for k, v in era5_land_data.items():
        #         era5_land_data[k] = _to_tensor(v if isinstance(v,np.ndarray) else v.values)

        # if self.era5_data and era5_data is not None:
        #     for k, v in era5_data.items():
        #         era5_data[k] = _to_tensor(v)

        if self.full_return and self.era5_data:
            return collated_imgs, era5_land_data, era5_data, paths
        elif self.era5_data:
            return collated_imgs, era5_land_data, era5_data
        elif self.full_return:
            return collated_imgs, paths
        return collated_imgs

    def plot(self, sample, num_samples=None):
        B = sample[list(sample.keys())[0]].shape[0] if num_samples is None else num_samples

        fig, ax = plt.subplots(B, len(self.products), figsize=(len(self.products) * 4, B * 4), squeeze=False)

        min_values = [1] * len(self.products)
        max_values = [0] * len(self.products)

        for b in range(B):
            for i, product in enumerate(self.products):
                if product == "S2-10m":
                    bands = self.rgb_bands

                else:
                    bands = []
                    prod_bands = [band for band in self._get_product_bands(product) if band in sample]
                    if len(prod_bands) == 0:
                        continue

                    if len(prod_bands) == 2:
                        prod_bands = [prod_bands[0], prod_bands[1], prod_bands[1]]
                    elif len(prod_bands) > 3:
                        prod_bands = prod_bands[:3]

                    for prod_band_name in prod_bands:
                        if product != self.product_changes[product]:
                            product_name, product_gsd = product.split("-")
                            product_gsd = product_gsd.replace("m", "")
                            # S1 product has the GSD in the band name, need to update if we resample
                            if product_name == "S1":
                                prod_band_name = prod_band_name.replace(
                                    product_gsd, str(self.GSD(self.product_changes[product]))
                                )
                        bands.append(prod_band_name)

                if bands[0] not in sample:
                    continue

                image = torch.stack([sample[band][b] for band in bands])

                image_min = torch.min(image).item()
                image_max = torch.max(image).item()
                min_values[i] = min(image_min, min_values[i])
                max_values[i] = max(image_max, max_values[i])

                image = torch.squeeze(image, dim=(1)).cpu().numpy()
                image = image.transpose(1, 2, 0)
                # print("plotting image", product)
                # print(f"image stats, min: {image_min} max: {image_max}")

                image = np.ma.masked_invalid(image)

                # calculate 99th percentile
                image_min_p = np.percentile(image, 1)
                image_max_p = np.percentile(image, 99)

                image = (image - image_min_p) / (image_max_p - image_min_p)

                ax[b, i].imshow(image)
                ax[b, i].set_title(
                    f"{product} min: {min_values[i]:.2f}, max: {max_values[i]:.2f} \n ({', '.join(b.split(':')[-1] for b in bands)})"
                )

        plt.show()

        return fig