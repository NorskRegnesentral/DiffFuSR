import os
import torch
from torch import nn
import pytorch_lightning as pl

# tasks
# - pretrain
# - finetune

# more detailed tasks
# - classification
# - regression
# - segmentation
# - object detection
# - instance segmentation
# - self-supervised

class TaskRegistry:
    def __init__(self):
        self.tasks = {}

    def _register(self, task_name: str, task: pl.LightningModule):
        
        # do some instance checking

        if task_name is None:
            task_name = task.__name__

        # print(f"Registering task {task_name} with {task}")

        if task_name in self.tasks:
            raise ValueError(f"Task {task_name} already registered")
        
        self.tasks[task_name] = task

    def register(self, task_name: str = None, task_type: str = 'pretrain', task: pl.LightningModule = None):

        def _register_wrapper(task):
            self._register(task_name, task)
            return task

        return _register_wrapper

    def get_task(self, task_name: str) -> pl.LightningModule:
        return self.tasks[task_name]
    
    def build(self, task_cfg) -> pl.LightningModule: # TODO: pytorch lightning module
        task_name = task_cfg.get('task')
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found in registry, available tasks: {self.tasks}")
        task = self.get_task(task_name)
        return task(task_cfg)


TASKS = TaskRegistry()