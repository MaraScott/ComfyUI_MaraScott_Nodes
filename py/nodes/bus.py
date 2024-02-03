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

import torch

class Marasit_Bus:
    def __init__(self):
        self.default_mask = torch.zeros(1, 1, 1024, 1024)  # Example default mask

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{},
            "optional": {
                "bus" : ("BUS",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    _INPUT_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT", "IMAGE", "MASK",)
    RETURN_TYPES = ("BUS", ) + _INPUT_TYPES
    _INPUT_NAMES = ("model", "clip", "vae", "positive", "negative", "latent", "image", "mask")
    RETURN_NAMES = ("bus",) + _INPUT_NAMES
    FUNCTION = "bus_fn"
    CATEGORY = "Maras IT/Utilities"
    
    def bus_fn(self, **kwargs):
        # Initialize the bus tuple with None values for each parameter
        bus = kwargs.get('bus', (None,) * len(self._INPUT_NAMES))
        if len(bus) != len(self._INPUT_NAMES):
            raise ValueError("The 'bus' tuple must have the same number of elements as '_INPUT_NAMES'")

        outputs = {}
        for name, bus_value in zip(self._INPUT_NAMES, bus):
            _input = kwargs.get(name, bus_value)
            outputs[name] = self._determine_output_value(name, _input, bus_value)

        self._ensure_required_parameters(outputs)
        self._handle_special_parameters(outputs)

        # Prepare and return the output bus tuple with updated values
        out_bus = tuple(outputs[name] for name in self._INPUT_NAMES)
        return (out_bus,) + out_bus

    def _determine_output_value(self, name, _input, bus_value):
        if name in ('image', 'mask') and isinstance(_input, torch.Tensor):
            return _input if _input.nelement() > 0 and (name != 'mask' or _input.any()) else bus_value
        return _input if _input is not None else bus_value

    def _ensure_required_parameters(self, outputs):
        for param in ('model', 'clip', 'vae'):
            if not outputs.get(param):
                raise ValueError(f'Either "{param}" or bus containing a {param} should be supplied')

    def _handle_special_parameters(self, outputs):
        if outputs.get('mask') is None or torch.numel(outputs['mask']) == 0:
            outputs['mask'] = self.default_mask