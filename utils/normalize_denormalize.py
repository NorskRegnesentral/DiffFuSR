
import torch
import torch.nn.functional as F


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
