import pandas as pd
data_lpips = {
    "Model": [
        "ldm_baseline",
        "opensrmodel",
        "satlas",
        "sr4rs",
        "superimage",
        "DiffFuSR (NAIP Harm)",
        "DiffFuSR (NAIP Unharm)",
        "DiffFuSR (Worldstrat)",
    ],
    "Reflectance": [
        0.1239,
        0.0076,
        0.1197,
        0.0979,
        0.0068,
        0.0069,
        0.0109,
        0.0075,
    ],
    "Spectral": [
        12.8441,
        1.9739,
        15.1521,
        22.4905,
        1.8977,
        1.6738,
        2.0734,
        1.7327,
    ],
    "Spatial": [
        0.0717,
        0.0118,
        0.2766,
        1.0099,
        0.0004,
        0.0060,
        0.0061,
        0.0042,
    ],
    "Synthesis": [
        0.0409,
        0.0194,
        0.0648,
        0.0509,
        0.0130,
        0.0237,
        0.0245,
        0.0205,
    ],
    "Hallucination": [
        0.4558,
        0.0642,
        0.5999,
        0.3417,
        0.0357,
        0.1149,
        0.1434,
        0.0752,
    ],
    "Omission": [
        0.3558,
        0.6690,
        0.0588,
        0.1924,
        0.8844,
        0.3466,
        0.2846,
        0.6216,
    ],
    "Improvement": [
        0.1884,
        0.2668,
        0.3413,
        0.4659,
        0.0800,
        0.5383,
        0.5720,
        0.3032,
    ],
}

data_clip = {
    "Model": [
        "ldm_baseline",
        "opensrmodel",
        "satlas",
        "sr4rs",
        "superimage",
        "DiffFuSR (NAIP Harm)",
        "DiffFuSR (NAIP Unharm)",
        "DiffFuSR (Worldstrat)",
    ],
    "Reflectance": [
        0.1239,
        0.0076,
        0.1197,
        0.0979,
        0.0068,
        0.0069,
        0.0109,
        0.0075,
    ],
    "Spectral": [
        12.8441,
        1.9739,
        15.1521,
        22.4905,
        1.8977,
        1.6739,
        2.0734,
        1.7327,
    ],
    "Spatial": [
        0.0717,
        0.0118,
        0.2766,
        1.0099,
        0.0004,
        0.0061,
        0.0061,
        0.0042,
    ],
    "Synthesis": [
        0.0409,
        0.0194,
        0.0648,
        0.0509,
        0.0130,
        0.0238,
        0.0245,
        0.0205,
    ],
    "Hallucination": [
        0.5963,
        0.1052,
        0.6996,
        0.3099,
        0.0610,
        0.2883,
        0.3272,
        0.2612,
    ],
    "Omission": [
        0.2327,
        0.6927,
        0.0872,
        0.3486,
        0.8524,
        0.3358,
        0.2929,
        0.4646,
    ],
    "Improvement": [
        0.1710,
        0.2021,
        0.2132,
        0.3415,
        0.0866,
        0.3759,
        0.3799,
        0.2742,
    ],
}
# Define the mean values for each model and metric (ignoring the ± uncertainty)
data_nd = {
    "Model": [
        "ldm_baseline",
        "opensrmodel",
        "satlas",
        "sr4rs",
        "superimage",
        "DiffFuSR (NAIP Harm)",
        "DiffFuSR (NAIP Unharm)",
        "DiffFuSR (Worldstrat)",
    ],
    "Reflectance": [
        0.0505,
        0.0031,
        0.0489,
        0.0396,
        0.0029,
        0.0024,
        0.0038,
        0.0026,
    ],
    "Spectral": [
        9.6923,
        1.2632,
        12.1231,
        3.4044,
        1.5672,
        1.1103,
        1.2320,
        1.3277,
    ],
    "Spatial": [
        0.0715,
        0.0114,
        0.2742,
        1.0037,
        0.0132,
        0.0052,
        0.0053,
        0.0032,
    ],
    "Synthesis": [
        0.0285,
        0.0068,
        0.0227,
        0.0177,
        0.0046,
        0.0083,
        0.0086,
        0.0072,
    ],
    "Hallucination": [
        0.6067,
        0.3431,
        0.8004,
        0.7274,
        0.2026,
        0.4580,
        0.4927,
        0.3915,
    ],
    "Omission": [
        0.3088,
        0.4593,
        0.1073,
        0.1637,
        0.6288,
        0.3543,
        0.3373,
        0.4248,
    ],
    "Improvement": [
        0.0845,
        0.1976,
        0.0923,
        0.1089,
        0.1686,
        0.1876,
        0.1701,
        0.1837,
    ],
}

data = data_lpips.copy()  # Start with the LPIPS data
# Create DataFrame
df = pd.DataFrame(data).set_index("Model")

# Specify which metrics should be ranked ascending (lower is better) vs. descending (higher is better)
ascending_metrics = ["Reflectance", "Spectral", "Spatial", "Hallucination", "Omission"]
descending_metrics = ["Synthesis", "Improvement"]

# Compute ranks for each metric
ranks = pd.DataFrame(index=df.index)
for col in ascending_metrics:
    # For "↓" metrics: lower values get better (rank 1 = smallest)
    ranks[col + "_rank"] = df[col].rank(method="average", ascending=True)

for col in descending_metrics:
    # For "↑" metrics: higher values get better (rank 1 = largest)
    ranks[col + "_rank"] = df[col].rank(method="average", ascending=False)

# Compute the average rank across all metrics
ranks["Avg_Rank"] = ranks.mean(axis=1)

# Round Avg_Rank to three decimal places for readability
ranks["Avg_Rank"] = ranks["Avg_Rank"].round(3)

# Display the resulting ranks
print(ranks[["Avg_Rank"]])

