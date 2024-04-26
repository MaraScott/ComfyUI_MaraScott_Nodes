#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Upscaler.  Upscale a picture by 2 using a 9 Square Grid to upscale the visual in 9 sequences
#
###

import torch
import comfy

from ..inc.lib.image import Image
from .KSamplerNode import common_ksampler

from ..utils.log import *

class UpscalerGridNode:
    
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {"id":"UNIQUE_ID"},
            "required":{
                "image": ("IMAGE",),
                "scale_by": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),     
                "upscale_method": (cls.upscale_methods ,),           

                "model": ("MODEL",),
                "vae": ("VAE",),

                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),

                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                # "latent_image": ("LATENT", ),

                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),                

            },
            "optional": {
            }
        }

    RETURN_TYPES = (
        "IMAGE", 
        "IMAGE", 
        # "MASK", 
        "STRING"
    )
    
    RETURN_NAMES = (
        "upscaled_image", 
        "image", 
        # "mask", 
        "info", 
    )
    
    OUTPUT_NODE = True
    CATEGORY = "MarasIT/upscaling"
    DESCRIPTION = "An \"UPSCALER\" Node"
    FUNCTION = "fn"
    
    def fn(self, **kwargs):
        
        # Initialize the bus tuple with None values for each parameter
        image = kwargs.get('image', None)
        scale_by = kwargs.get('scale_by', None)
        upscale_method = kwargs.get('upscale_method', None)
        
        vae = kwargs.get('vae', None)
        model = kwargs.get('model', None)
        seed = kwargs.get('seed', None)
        steps = kwargs.get('steps', None)
        cfg = kwargs.get('cfg', None)
        sampler_name = kwargs.get('sampler_name', None)
        scheduler = kwargs.get('scheduler', None)
        positive = kwargs.get('positive', None)
        negative = kwargs.get('negative', None)
        denoise = kwargs.get('denoise', None)
        
        output_info = [f"No info"]
        
        if image is None:
            raise ValueError("UpscalerGridNode id XX: No image provided")

        if not isinstance(image, torch.Tensor):
            raise ValueError("UpscalerGridNode id XX: Image provided is not a Tensor")
        
        image_width = image.shape[2]
        image_height = image.shape[1]
        image_divisible_by_8 = Image.is_divisible_by_8(image)
        if not image_divisible_by_8:
            image_divisible_by_8 = False
            image_width, image_height = Image.calculate_new_dimensions(image_width, image_height)

        image = Image.upscale(image, upscale_method, image_width, image_height, "center")[0]
            
        # _mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        
        upscaled_image = Image.upscaleBy(image, upscale_method, scale_by)[0]

        grid_images = Image.get_grid_images(upscaled_image)
        
        output_images = []
        for grid_image in grid_images:            
            # Encode the upscaled image using the VAE
            t = vae.encode(grid_image[:,:,:,:3])
            latent_image = {"samples":t}
            
            # Use the latent image in the common_ksampler function
            latent_output = common_ksampler(
                model, 
                seed, 
                steps, 
                cfg, 
                sampler_name, 
                scheduler, 
                positive, 
                negative, 
                latent_image, 
                denoise
            )[0]
            output = vae.decode(latent_output["samples"]).unsqueeze(0)
            
            # Collect all outputs (you may want to adjust this depending on how you want to handle the outputs)
            output_images.append(output)

        output_image = Image.rebuild_image_from_parts(output_images, upscaled_image)
                
        output_image_width = output_image.shape[2]
        output_image_height = output_image.shape[1]

        output_info = [f"""

IMAGE (INPUT)
    width   :   {image_width}
    height  :   {image_height}
    image divisible by 8 : {image_divisible_by_8}

------------------------------

IMAGE (OUTPUT)
    width   :   {output_image_width}
    height  :   {output_image_height}

"""]
        
        return (
            output_image,
            image,
            # output_mask,
            output_info
        )
