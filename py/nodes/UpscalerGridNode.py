#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Upscaler.  Upscale a picture by 2 using a 9 Square Grid to upscale the visual in 9 sequences
#
###

import torch
import numpy as np

from ..inc.lib.image import Image

from ..utils.log import *

class UpscalerGridNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {"id":"UNIQUE_ID"},
            "required":{
                "image": ("IMAGE",),
            },
            "optional": {
            }
        }

    RETURN_TYPES = (
        "IMAGE", 
        # "MASK", 
        "STRING"
    )
    
    RETURN_NAMES = (
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

        upscale_method = "nearest-exact"

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
            image = Image.upscale(image, upscale_method, image_width, image_height, True)[0]
            
        _image = image
        # _mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

        scale_by = 2
        _image = Image.upscaleBy(_image, upscale_method, scale_by)[0]
        
        # Divide the upscaled image into 9 parts
        grid_images = Image.get_grid_images(_image)
        log(grid_images)
        output_images = [Image.upscaleBy(part[0], "nearest-exact", 2) for part in grid_images]

        output_image = Image.rebuild_image_from_parts(output_images, _image.shape[2], _image.shape[1])[0]
        
        # output_image = _image
        # output_mask = _mask.unsqueeze(0)                
        
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
            # output_mask,
            output_info
        )
