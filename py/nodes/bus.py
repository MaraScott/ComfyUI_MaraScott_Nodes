#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Bus.  Converts X connectors into one, and back again.  You can provide a bus as input
#       or the X separate inputs, or a combination.  If you provide a bus input and a separate
#       input (e.g. a model), the model will take precedence.
#
#       The term 'bus' comes from computer hardware, see https://en.wikipedia.org/wiki/Bus_(computing)
#       Largely inspired by Was Node Suite - Bus Node 
#
###

import os
import json
import torch

# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
ANY = AnyType("*")

class Bus_node:
    def __init__(self):
        self.default_mask = torch.zeros(1, 1, 1024, 1024)  # Example default mask

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{},
            "optional": {
                # "bus" : ("BUS",),
                # "pipe" : ("BASIC_PIPE",),
                # "model": ("MODEL",),
                # "clip": ("CLIP",),
                # "vae": ("VAE",),
                # "positive": ("CONDITIONING",),
                # "negative": ("CONDITIONING",),
                # "latent": ("LATENT",),
                # "image": ("IMAGE",),
                # "mask": ("MASK",),
                # "any": (ANY, {}),
            }
        }

    # _INPUT_TYPES = ("BASIC_PIPE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT", "IMAGE", "MASK", ANY,)
    # _INPUT_TYPES = ()
    # RETURN_TYPES = ("BUS",) + _INPUT_TYPES
    RETURN_TYPES = ()
    # _INPUT_NAMES = ("basic_pipe", "model", "clip", "vae", "positive", "negative", "latent", "image", "mask", "any",)
    # _INPUT_NAMES = ()
    # RETURN_NAMES = ("bus",) + _INPUT_NAMES
    RETURN_NAMES = ()
    FUNCTION = "bus_fn"
    CATEGORY = "MarasIT/utils"
    DESCRIPTION = "A Universal Bus/Pipe Node"
    
    def bus_fn(self, **kwargs):
        
        profile = 'default'
        inputsByNode = os.getenv(f"MarasITBusNode")
        inputsByNode = json.loads(inputsByNode)
        print(inputsByNode)
        # Initialize the bus tuple with None values for each parameter
        inputs = {}
        for name in inputsByNode:
            print(name)
            if name.startswith('profile_'):
                profile = name.split("profile_")[-1]
            if name != 'bus' and name != 'pipe' and not name.startswith('profile_'):
                inputs[name] = None
        outputs = inputs.copy()
        in_bus = kwargs.get('bus', (None,) * len(inputs))

        # Update outputs based on in_bus values
        for i, name in enumerate(inputs):
            if in_bus[i] is not None:  # Only update if in_bus value is not None
                outputs[name] = in_bus[i]
                
        in_pipe = kwargs.get('pipe', (None,) * len(inputs))
        # Update outputs based on in_bus values
        for i, name in enumerate(inputs):
            if in_pipe[i] is not None:  # Only update if in_bus value is not None
                outputs[name] = in_pipe[i]

        # Update outputs based on inputs and current outputs
        for name, value in inputs.items():
            inputs[name] = kwargs.get(name, None)
            outputs[name] = self._determine_output_value(name, inputs[name], outputs[name])

        # self._ensure_required_parameters(outputs)
        self._handle_special_parameters(outputs)

        # Prepare and return the output bus tuple with updated values
        out_bus = tuple(outputs[name] for name in outputs)
        return (out_bus,) + (out_bus,) + out_bus


    def _determine_output_value(self, name, _input, value):
        if name in ('image', 'mask') and isinstance(_input, torch.Tensor):
            return _input if _input.nelement() > 0 and (name != 'mask' or _input.any()) else value
        return _input if _input is not None else value

    def _ensure_required_parameters(self, outputs):
        for param in ('model', 'clip', 'vae'):
            if not outputs.get(param):
                raise ValueError(f'Either "{param}" or bus containing a {param} should be supplied')

    def _handle_special_parameters(self, outputs):
        if outputs.get('mask') is None or torch.numel(outputs['mask']) == 0:
            outputs['mask'] = self.default_mask