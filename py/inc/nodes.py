#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

import torch

class AnyType(str):
    """Hack: String type that is always equal in not equal comparisons"""
    def __ne__(self, __value: object) -> bool:
        return False

class Configuration:
    OUTPUT_NODE = False
    CATEGORY = "MaraScott/utils"
    # SHAPE = LiteGraph.CARD_SHAPE
    # COLOR = "#8154A6"
    # BGCOLOR = "#8154A6"
    # GROUPCOLOR = "#8154A6"    

    DEFAULT = {
        "mask": torch.zeros(1, 1, 1024, 1024)  # Example default mask
    }

    TYPES = {
        "ANY": AnyType("*"),  # Our any instance wants to be a wildcard string
    }

    @classmethod
    def generate_entries(cls, input_names, input_types, code='py'):
        """Generate entries for node configuration"""
        entries = {}
        for name, type_ in zip(input_names, input_types):
            if code == 'js':
                entries[name] = type_
            else:
                if name.startswith("text"):
                    entries[name] = (type_, {"forceInput": True, "multiline": True})
                elif name in {"width", "height"}:
                    entries[name] = (type_, {"default": 1024, "min": 1, "max": 8192, "forceInput": True})
                else:
                    entries[name] = (type_,)
        return entries

    @staticmethod
    def determine_output_value(name='', input_value=None, preset_value=None):
        """Determine the output value for a given input"""
        if name in {'image', 'mask'} and isinstance(input_value, torch.Tensor):
            return input_value if input_value.nelement() > 0 and (name != 'mask' or input_value.any()) else preset_value
        if name.startswith('text'):
            return input_value if input_value is not None else preset_value if preset_value is not None else ''
        return input_value if input_value is not None else preset_value

    @staticmethod
    def ensure_required_parameters(outputs):
        """Ensure required parameters are present in the outputs"""
        for param in {'model', 'clip', 'vae'}:
            if not outputs.get(param):
                raise ValueError(f'Either "{param}" or bus containing a {param} should be supplied')

    @staticmethod
    def handle_special_parameters(outputs):
        """Handle special parameters in the outputs"""
        if outputs.get('mask') is None or torch.numel(outputs['mask']) == 0:
            outputs['mask'] = Configuration.DEFAULT["mask"]

    @staticmethod
    def get_kwarg_with_prefix(kwargs, prefix, default_value=None):
        """Get the first kwarg that starts with the given prefix"""
        filtered_kwargs = {key: value for key, value in kwargs.items() if key.startswith(prefix)}
        return next(iter(filtered_kwargs.values()), default_value)
