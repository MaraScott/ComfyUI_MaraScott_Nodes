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
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("BUS", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT", "IMAGE", "MASK",)
    RETURN_NAMES = ("bus", "model", "clip", "vae", "positive", "negative", "latent", "image", "mask")
    FUNCTION = "bus_fn"
    CATEGORY = "Maras IT/Utilities"

    def bus_fn(self, bus=(None,None,None,None,None,None,None,None), model=None, clip=None, vae=None, positive=None, negative=None, latent=None, image=None, mask=None):

        # Unpack the 5 constituents of the bus from the bus tuple.
        (bus_model, bus_clip, bus_vae, bus_positive, bus_negative, bus_latent, bus_image, bus_mask) = bus

        # If you pass in specific inputs, they override what comes from the bus.
        out_model       = model     or bus_model
        out_clip        = clip      or bus_clip
        out_vae         = vae       or bus_vae
        out_positive    = positive  or bus_positive
        out_negative    = negative  or bus_negative
        out_latent      = latent    or bus_latent
        
        # Check and handle 'image' input
        if image is not None and image.numel() > 0:
            out_image = image
        else:
            out_image = bus_image

        # Check and handle 'mask' input
        if mask is not None and torch.any(mask):
            out_mask = mask
        else:
            out_mask = bus_mask

        # Squash all 5 inputs into the output bus tuple.
        out_bus = (out_model, out_clip, out_vae, out_positive, out_negative, out_latent, out_image, out_mask)

        if not out_model:
            raise ValueError('Either model or bus containing a model should be supplied')
        if not out_clip:
            raise ValueError('Either clip or bus containing a clip should be supplied')
        if not out_vae:
            raise ValueError('Either vae or bus containing a vae should be supplied')
        if not out_mask:
            out_mask = torch.zeros(1, 1, 1024, 1024)

        # We don't insist that a bus contains conditioning.

        return (out_bus, out_model, out_clip, out_vae, out_positive, out_negative, out_latent, out_image, out_mask)
    