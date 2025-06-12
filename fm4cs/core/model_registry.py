# Copyright (c) 2024 Stanford Machine Learning Group

import re

import torch
from torch import nn

from fm4cs.utils.helper import extract_model_state_dict_from_ckpt
from fm4cs.utils.patch_embed import pi_resize_patch_embed
from fm4cs.utils.pos_embed import interpolate_pos_embed_usat


class ModelRegistry:
    def __init__(self):
        self.models = {}

    def _register(self, model_name: str, model: nn.Module):

        # do some instance checking

        if model_name is None:
            model_name = model.__name__

        if model_name in self.models:
            raise ValueError(f"Model {model_name} already registered")

        self.models[model_name] = model

    def register(self, model_name: str = None, model: nn.Module = None):

        def _register_wrapper(model):
            self._register(model_name, model)
            return model

        return _register_wrapper

    def get_model(self, model_name: str) -> nn.Module:
        return self.models[model_name]

    def build(self, model_cfgs) -> nn.Module:

        if model_cfgs.get("name", None) is not None: # single model config
            model_cfgs = {model_cfgs["name"]: model_cfgs}

        models = {}

        for model_name, model_cfg in model_cfgs.items():

            model_type = model_cfg.get("type")

            if model_type not in self.models:
                raise ValueError(f"Model {model_name} not found in registry, available models: {self.models}")

            input_params = model_cfg.get("input_params", {})
            model_kwargs = model_cfg.get("kwargs", {})

            channels = input_params.get("channels", None)
            ground_cover = input_params.get("ground_cover", None)

            def _get_patch_size(ground_cover: int, GSD: int, num_patch: int) -> tuple[int, int]:
                patch_size = int(ground_cover / (GSD * num_patch))
                assert patch_size * GSD * num_patch == ground_cover, f"Patch size {patch_size} does not divide ground cover {ground_cover} evenly"
                return patch_size, patch_size

            print(f"Building model {model_name} with input params: {input_params}, kwargs {model_kwargs}")

            model = self.get_model(model_type)(input_params, **model_kwargs)

            ckpt = model_cfg.get("ckpt", None)
            ckpt_ignore = model_cfg.get("ckpt_ignore", [])
            ckpt_copy = model_cfg.get("ckpt_copy", [])
            ckpt_remap = model_cfg.get("ckpt_remap", {})
            strict = model_cfg.get("strict", True)
            resize_patch_embed = model_cfg.get("resize_patch_embed", False)
            target_model = model_cfg.get("target_model", model_name)

            if ckpt is not None:
                print(f"Loading custom weight for {model_name} from {ckpt}")
                # NOTE: downside, it will try to load ckpt every single time.
                # but support loadingt different from ckpt for different modules
                ckpt = torch.load(ckpt, map_location="cpu")
                model_state_dict = extract_model_state_dict_from_ckpt(ckpt)[target_model]

                # Pop all of the keys that can be ignored
                new_keys = list(model_state_dict.keys())
                for rgx_item in ckpt_ignore:
                    re_expr = re.compile(rgx_item)
                    new_keys = [key for key in new_keys if not re_expr.match(key)]
                model_state_dict = {k: model_state_dict[k] for k in new_keys}

                # Add in all keys you want to copy the default param for
                for copy_key in ckpt_copy:
                    print(f"Skipping model load for: {copy_key}")
                    model_state_dict[copy_key] = model.state_dict()[copy_key]

                # Remap certain keys for multi sensor pretrain to single sensor finetune
                for key, cfg_map in ckpt_remap.items():
                    print(f"Remapping key for custom load: {key}")
                    old_val = model_state_dict.pop(key)
                    new_val = old_val
                    # Apply modifications
                    new_name = cfg_map.get("name", key)
                    params = cfg_map.get("params", {})
                    func = cfg_map.get("func", None)
                    if func == "index_select":
                        if isinstance(params["indices"], str):
                            start,stop = params["indices"].split(":")
                            params["indices"] = torch.arange(int(start), int(stop))
                        new_val = torch.index_select(old_val, params["dim"], torch.tensor(params["indices"]))
                    elif func == "concat_passthrough":
                        new_val = torch.cat((new_val, torch.index_select(model.state_dict()[new_name], params["dim"], torch.tensor(params["index"]))), dim=params["dim"])
                    # elif func == "replace":
                    #     new_val = model.state_dict()[new_name]
                    # elif func == "resize":
                    #     new_val = pi_resize_patch_embed(old_val, params["new_patch_size"])

                    model_state_dict[new_name] = new_val

                if resize_patch_embed:
                    patch_embed_keys = [k for k in model_state_dict.keys() if "patch_embed" in k and "weight" in k]

                    for patch_embed_key in patch_embed_keys:
                        channel_key = patch_embed_key.split(".")[-2]
                        new_patch_size = _get_patch_size(ground_cover, **channels[channel_key])

                        if model_state_dict[patch_embed_key].shape[2:] != new_patch_size:
                            print(f"Resizing patch embed for {patch_embed_key}")
                            model_state_dict[patch_embed_key] = pi_resize_patch_embed(model_state_dict[patch_embed_key], new_patch_size)

                # Interpolate pos embedding if necessary
                pos_embed_keys = [f"pos_embed.{k}" for k in model.pos_embed.keys()]
                ref_pos_embed_key = sorted(pos_embed_keys, key=lambda x: int(x.split(".")[-1]))[0]
                pos_embeds_needs_reinit = False
                if ref_pos_embed_key in model_state_dict and  model.state_dict()[ref_pos_embed_key].shape != model_state_dict[ref_pos_embed_key].shape:
                    print("interpolating pos_embed")

                    # interpolating the ref pos embed (lowest GSD)
                    interpolate_pos_embed_usat(model, model_state_dict, ref_pos_embed_key)

                    # remove the other pos embeds, as they will be reinitialized afterwards
                    for k in pos_embed_keys:
                        if k in model_state_dict:
                            del model_state_dict[k]

                    pos_embeds_needs_reinit = True

                # TODO: fix fuzzy grid size
                model.load_state_dict(model_state_dict, strict=strict)
                print(f"Custom weight loaded for {model_name}")

                if pos_embeds_needs_reinit:
                    model.init_embeds(pos_only=True)

            models[model_name] = model

        return models

# global model registry
MODELS = ModelRegistry()
