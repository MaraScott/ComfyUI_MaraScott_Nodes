#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# File: __init__.py
# Project: ComfyUI-MarasIT-Nodes
# By MarasIT (Discord: davask#4370)
# Copyright 2024 David Asquiedge (davask)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os 
import sys 

import folder_paths as comfy_paths

p310_plus = (sys.version_info >= (3, 10))

MANIFEST = {
    "name": "Maras IT Nodes",
    "version": (1,0,0),
    "author": "davask",
    "project": "https://github.com/davask/ComfyUI-MarasIT-Nodes",
    "description": "A simple Bus node",
}

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "marasit_nodes_comfyui"))
sys.path.append(comfy_paths.base_path)


# Bus.  Converts X connectors into one, and back again.  You can provide a bus as input
#       or the X separate inputs, or a combination.  If you provide a bus input and a separate
#       input (e.g. a model), the model will take precedence.
#
#       The term 'bus' comes from computer hardware, see https://en.wikipedia.org/wiki/Bus_(computing)
#       Largely inspired by Was Node Suite - Bus Node 
    
class Marasit_Bus:
    def __init__(self):
        pass

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
            }
        }
    RETURN_TYPES = ("BUS", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT", "IMAGE",)
    RETURN_NAMES = ("bus", "model", "clip", "vae", "positive", "negative", "latent", "image")
    FUNCTION = "bus_fn"
    CATEGORY = "Maras IT/Utilities"

    def bus_fn(self, bus=(None,None,None,None,None,None,None), model=None, clip=None, vae=None, positive=None, negative=None, latent=None, image=None):

        # Unpack the 5 constituents of the bus from the bus tuple.
        (bus_model, bus_clip, bus_vae, bus_positive, bus_negative, bus_latent, bus_image) = bus

        # If you pass in specific inputs, they override what comes from the bus.
        out_model       = model     or bus_model
        out_clip        = clip      or bus_clip
        out_vae         = vae       or bus_vae
        out_positive    = positive  or bus_positive
        out_negative    = negative  or bus_negative
        out_latent      = latent    or bus_latent
        out_image       = image     or bus_image

        # Squash all 5 inputs into the output bus tuple.
        out_bus = (out_model, out_clip, out_vae, out_positive, out_negative, out_latent, out_image)

        if not out_model:
            raise ValueError('Either model or bus containing a model should be supplied')
        if not out_clip:
            raise ValueError('Either clip or bus containing a clip should be supplied')
        if not out_vae:
            raise ValueError('Either vae or bus containing a vae should be supplied')
        # We don't insist that a bus contains conditioning.

        return (out_bus, out_model, out_clip, out_vae, out_positive, out_negative, out_latent, out_image)
    


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MarasitBusNode": Marasit_Bus,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarasitBusNode": "Bus Node (By Maras IT)"
}