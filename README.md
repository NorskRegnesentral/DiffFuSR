# DiffFuSR: Super-Resolution of all Sentinel-2 Multispectral Bands using Diffusion Models



> _Train and evaluate RGB SR models, multispectral fusion networks, and OpenSR metrics._
> https://arxiv.org/abs/2506.11764

---

## 0. Environment setup 

```bash
# ❶ Create a virtual environment (optional but recommended) Python version used 3.11.4
python -m venv .venv
source .venv/bin/activate          # Windows: venv\Scripts\activate

# ❷ Install all Python dependencies
pip install -r requirements.txt
```

To directly download and test our pretrained checkpoints, skip diretly to Step 2a.

## 1. Train the Super-Resolution (SR) models

### 1 a. Prepare training data

| Dataset       | What to download                                                                 | Destination                     |
|---------------|----------------------------------------------------------------------------------|---------------------------------|
| NAIP Synthetic | *.tif tiles from https://huggingface.co/datasets/isp-uv-es/SEN2NAIP/tree/main/synthetic | load/naip/synthetic_naip/      |
| WorldStrat    | *.tif files from https://worldstrat.github.io/                                   | load/WorldStrat_raw/           |

Make sure only TIFF files are placed in the folders above.

### 1 b. Run the training script three times

```bash
python 1_train_sr.py --config configs/blindsrsnf_aniso_naip_degraded_harm_large.yaml
python 1_train_sr.py --config configs/blindsrsnf_aniso_naip_degraded_not_harm_large.yaml
python 1_train_sr.py --config configs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large.yaml
```

Each run will create a Lightning log directory such as `logs/blindsrsnf_aniso_naip_degraded_harm_large/`.

## 2. Test the SR models on OpenSR-Test

### 2 a. Download the test set



```bash
python download_opensr_test.py
```

### 2 b. Split Gather all LR/HR pairs to a common folder
It will use lr_files_list.txt to dump all low and high resolution images.

```bash
python 0_prepare_open_sr_test_data.py 
```

### 2 c. Place pretrained checkpoints
The following checkpoints are available:
- WorldStrat SR
- NAIP-no-harm SR (No Harmonization)
- NAIP-harm SR (with harmonization)
Checkpoints are stored at [https://huggingface.co/NorskRegnesentralSTI/DiffFuSR](https://huggingface.co/NorskRegnesentralSTI/DiffFuSR). Download them using:

```bash
git lfs install
git clone https://huggingface.co/NorskRegnesentralSTI/DiffFuSR && mv DiffFuSR/logs logs
```

### 2 d. Run the RGB test script needed for open sr test

```bash
python 2_test_rgb_for_opensr_metric.py
```
`--checkpoint` flag can be changed to test all three models. WorldStrat SR, NAIP-no-harm SR (No Harmonization) and  NAIP-harm SR (with harmonization).
SR outputs are saved under the corresponding `logs/.../sr/` directory.

### 2 e. Re-package results for Open SR metric computation
it will produce folder diffsr in logs folder which will be used in next step.
```bash
python 3_preprocess_for_opensrtest.py
```

### 2 f. Compute OpenSR metrics

Change the path_diffsr appropriately. This script will create the open sr metrics for the three SR evaluation tables (TABLE I to TABLE III). We follow open-SR test documentation to correctly set all experiment setting. These can be changed as required. 

e.g. path_diffsr = "logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/results/worldstrat/diffsr"

```bash
python 4_opensr.py 
```

The script writes a CSV file with OpenSR metrics. Open it in and process in Excel to obtain averages and variance across all test images. 

## 3. Train the Fusion (multispectral) model
The fusion model is trained on the Norwegian Computing Center FM4CS dataset, released at
[FM4CS/THOR-Pretrain](https://huggingface.co/datasets/FM4CS/THOR-Pretrain).
The full dataset is too large to download in full.

We provide the list of tiles used for training in `list_fusion_train.txt`.
If you build a better downloader that fetches only those tiles, a pull request is welcome.




Run:

```bash
python 5_train_fusion.py 
```

The model checkpoints are saved in `logs/GSD/`.

## 4. Test the complete SR + Fusion pipeline
Test using pretrained model using and the comlete super-resolutiona nd fusion pipeline. Select the correct flag desired. Either use Gram Schmidt or Neural Network for fusion. For neural network a pre-trained weight are required. Also chose the super-resolution model to be used.

Downloads weight for fusion module must be in `logs/GSD/`.

```bash
python 6_test_multispectral_SR_fuse.py 
 
```

## 5. Benchmark the full pipeline
to bechmark the whole SR + fusion pipeline. This will generate the Last table in the paper. TABLE IV
```bash
python 7_benchmark_DiffFuSR.py
```

The script prints all metrics

## Folder layout (summary)

```bash
load/
├─ naip/synthetic_naip/          # NAIP synthetic TIFFs
├─ WorldStrat_raw/               # WorldStrat raw TIFFs
└─ opensrtest/100/
   ├─ lr/                        # auto-generated
   └─ hr/                        # auto-generated

logs/
├─ blindsrsnf_aniso_naip_degraded_harm_large/
│  └─ version_1/checkpoints/last.ckpt
├─ blindsrsnf_aniso_naip_degraded_not_harm_large/
│  └─ version_0/checkpoints/last.ckpt
├─ blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/
│  └─ version_7/checkpoints/last.ckpt
└─ GSD/                          # Fusion model
   └─ best.ckpt
```

## Citation

If you use this pipeline, please cite our paper and also the works which this is based on.


Thanks to following sources for code inspiration.

https://github.com/hanlinwu/BlindSRSNF

https://github.com/ESAOpenSR/opensr-test

https://github.com/esaOpenSR/opensr-degradation/
