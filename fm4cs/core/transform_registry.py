# Copyright (c) 2024 Stanford Machine Learning Group

import typing as T

import torchvision.transforms as transforms

class PretrainTransformRegistry:
    def __init__(self):
        self.transforms = {}

    def _register(self, transform_name: str, transform, override: bool = False):
        
        # do some instance checking

        if transform_name is None:
            transform_name = transform.__name__

        if transform_name in self.transforms and not override:
            raise ValueError(f"transform {transform_name} already registered")
        
        self.transforms[transform_name] = transform

    def register(self, override: bool = False, transform_name: str = None, transform = None):

        def _register_wrapper(transform):
            self._register(transform_name, transform, override)
            return transform

        return _register_wrapper

    def get_transform(self, transform_name: str):
        return self.transforms[transform_name]
    
    ######### usat https://github.com/stanfordmlgroup/USat/blob/main/usat/utils/builder.py#L58 #####


def build(self, cfg: T.Dict[str, T.Any], target: str = 'transform1') -> T.Any:
    transform_cfg = cfg.get('pretrain_transform', None)
    if transform_cfg is None:
        raise KeyError(f"Provided cfg does not contain transform config.")
    
    target_cfg = transform_cfg.get(target, None)
    if target_cfg is None: 
        raise KeyError(f"Provided cfg does not contain {target} config.")
    
    transform_list  = []
    for transform_name, transform_args in target_cfg.items():
        transform_class = self.transforms.get(transform_name)

        if transform_class is None:
            raise KeyError(f'Provided key {transform_name} is not available. Registry has: ', self.transforms)
        random_apply_p = transform_args.pop('random_apply', None)
        if random_apply_p is None:
            transform_list.append(transform_class(**transform_args))
        else:
            transform_list.append(transforms.RandomApply([transform_class(**transform_args)], p=random_apply_p))


    return transforms.Compose(transform_list)




# global transform registry
PRETRAIN_TRANSFORMS = PretrainTransformRegistry()