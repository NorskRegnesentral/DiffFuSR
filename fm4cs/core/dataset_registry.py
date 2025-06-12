# Copyright (c) 2024 Stanford Machine Learning Group

import typing as T
import re
import os

import torch
from torch import nn
from torch.utils.data import Dataset
from .transform_registry import PRETRAIN_TRANSFORMS

class DatasetRegistry:
    def __init__(self):
        self.datasets = {}

    def _register(self, dataset_name: str, dataset: Dataset, override: bool = False):
        
        # do some instance checking

        if dataset_name is None:
            dataset_name = dataset.__name__

        if dataset_name in self.datasets and not override:
            raise ValueError(f"Dataset {dataset_name} already registered")
        
        self.datasets[dataset_name] = dataset

    def register(self, override: bool = False, dataset_name: str = None, dataset: Dataset = None):

        def _register_wrapper(dataset):
            self._register(dataset_name, dataset, override)
            return dataset

        return _register_wrapper

    def get_dataset(self, dataset_name: str) -> Dataset:
        return self.datasets[dataset_name]
    
    ######### usat https://github.com/stanfordmlgroup/USat/blob/main/usat/utils/builder.py#L58 #####

    def build(self, dataset_cfg: T.Dict[str, T.Any], split: str = 'train') -> nn.Module:
        
        dataset_kwargs = dataset_cfg.get('kwargs', {})
        
        print(f"Dataset config: {dataset_cfg}")
        #print(f'Available datasets: {self.datasets}')

        build_cfg = dataset_cfg.copy()
        dataset_name = build_cfg.pop("name")
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found in registry, available datasets: {self.datasets}")
        
        dataset = self.datasets.get(dataset_name)
        
        dataset_kwargs = build_cfg.get(f'{split}_kwargs', {})

        print(f'Building dataset {dataset_name} with kwargs: {dataset_kwargs}')

        if dataset_kwargs.get('custom_transform', None):
            transform_args = {'pretrain_transform':dataset_kwargs}
            custom_trans = PRETRAIN_TRANSFORMS.build(transform_args, target='custom_transform')
        else:
            custom_trans = None
        
        dataset_kwargs['custom_transform'] = custom_trans

        return dataset(**dataset_kwargs)

# global dataset registry
DATASETS = DatasetRegistry()