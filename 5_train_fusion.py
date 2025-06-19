import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from fm4cs.data.fm4cs_footprint_builder import FootprintBuilder
from fm4cs.data.fm4cs_iterable_dataset_v2 import FM4CSIterableDatasetV2
from fm4cs.model import FusionNetwork

seed = 42

GSD = 'GSD'

plot = False    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
dataset_cfg = {
    "paths": "/lokal-uten-backup-8tb/pro/fm4cs/usr/tforgaard/data/lumi/v2",
    "split": None,
    "ground_cover": 1440,  # use highest cover for combined mode
    "sample_ratio": 0.01,   # adjust as needed
    "resize": True,
    "include_filter": None,
    "exclude_filter": None,
    "important_products": ["S2"],
    "products": {
        "S2-10m": None,
        "S2-20m": None,
        "S2-60m": None,
        #"S3-250m": None,
    },
    "strict_data_mode": True,
}
dataset_cfg_val = {
    "paths": "/lokal-uten-backup-8tb/pro/fm4cs/usr/tforgaard/data/lumi/v2",
    "split": None,
    "ground_cover": 1440,
    "sample_ratio": 0.0001,
    "resize": True,
    "include_filter": None,
    "exclude_filter": None,
    "important_products": ["S2"],
    "products": {
        "S2-10m": None,
        "S2-20m": None,
        "S2-60m": None,
        #"S3-250m": None,
    },
    "strict_data_mode": True,
}


save_fig = False  # True to create and save footprint figures
overwrite = False  # True to overwrite existing metadata cache files
num_workers = 4

footprint_builder = FootprintBuilder(**dataset_cfg, save_fig=save_fig, overwrite=overwrite)

footprint_dataloader = DataLoader(
    dataset=footprint_builder,
    batch_size=1,
    shuffle=False,
    collate_fn=footprint_builder.collate_fn,
    num_workers=num_workers,
    drop_last=False,
)

misses = 0
misses_files = []
found_files = []

for sample in tqdm(footprint_dataloader, total=len(footprint_dataloader), desc="Building dataset"):
    file_path, gdf, total_area, success = sample[0]
    if not success:
        misses += 1
        misses_files.append(str(file_path))
    else:
        found_files.append(str(file_path))


print(f"Misses: {misses}, {misses / len(footprint_dataloader) * 100:.2f}%")
print(f"Misses files: {misses_files}")
print("===============================")
print(f"Found {len(found_files)} files")



# 

if plot:

    num_workers = 0
    batch_size = 4

    dataset = FM4CSIterableDatasetV2(**dataset_cfg, batch_size=batch_size)

    g = torch.Generator()
    g.manual_seed(seed)

    if num_workers == 0:
        seed_worker(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=1 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )

# train model 

model = FusionNetwork(mode='train',GSD=GSD)   
# Inputs
gpu_ids = [3]
log_every_n_steps = 1
val_every_n_steps = 1
max_epochs = 50
# add GSD to log path
tb_logger = loggers.TensorBoardLogger(f"logs/{GSD}")
num_workers = 4
batch_size = 4

val_checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="metric_val",
    mode="min",
    #dirpath="my/path/",
    filename=f"best-{GSD}-"+"{epoch:02d}-{metric_val:.3f}-{loss_train:.4f}",
)
# add GSD to filename

latest_checkpoint_callback = ModelCheckpoint(
            filename=f"latest-{GSD}-"+"{epoch:02d}-{metric_val:.3f}-{loss_train:.4f}", #f"latest-{GSD}-"+"{epoch:02d}-{loss_val:.3f}-{loss_train:.4f}",
            save_top_k=1,
            mode="max",
            monitor="step",
            every_n_epochs=log_every_n_steps,save_on_train_epoch_end=True
        )

callbacks = [val_checkpoint_callback, latest_checkpoint_callback]

# trainer = pl.Trainer(devices=gpu_ids,max_epochs=max_epochs,log_every_n_steps=log_every_n_steps,
#         check_val_every_n_epoch=val_every_n_steps,callbacks=callbacks,logger=tb_logger) #default_root_dir = default_root_dir,
trainer = pl.Trainer(
    accelerator="gpu",
    devices=gpu_ids,
    strategy="ddp_find_unused_parameters_true",  # Enable finding unused parameters
    max_epochs=max_epochs,
    log_every_n_steps=log_every_n_steps,
    check_val_every_n_epoch=val_every_n_steps,
    callbacks=callbacks,
    logger=tb_logger,
    sync_batchnorm=True,  # Synchronize batch normalization between GPUs
    num_nodes=1,  # Number of machines (use 1 for single machine)
)

dataset = FM4CSIterableDatasetV2(**dataset_cfg, batch_size=batch_size)
val_dataset = FM4CSIterableDatasetV2(**dataset_cfg_val, batch_size=1)  

# train_loader = DataLoader(dataset)
g = torch.Generator()
g.manual_seed(seed)

if num_workers == 0:
    seed_worker(seed)

train_loader = DataLoader(
    dataset,
    batch_size=None,
    collate_fn=dataset.collate_fn,
    shuffle=False,
    num_workers=num_workers,
    prefetch_factor=1 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,)
val_loader = DataLoader(
    val_dataset,
    batch_size=None,
    collate_fn=val_dataset.collate_fn,
    shuffle=False,
    num_workers=num_workers,
    prefetch_factor=1 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader) #, val_dataloaders=val_loader

