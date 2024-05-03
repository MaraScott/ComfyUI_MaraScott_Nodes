#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Upscaler Refiner Node.  Upscale and Refine a picture by 2 using a 9 Square Grid to upscale and refine the visual in 9 sequences
#
###

import torch
import comfy
import comfy_extras
import comfy_extras.nodes_custom_sampler
from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
import nodes
import folder_paths

from ...utils.version import VERSION

from ...inc.lib.image import Image

from ...utils.log import *

class UpscalerRefiner_McBoaty:
    
    SIGMAS_TYPES = [
        "BasicScheduler"
        , "SDTurboScheduler"
        , "AlignYourStepsScheduler"
    ]
    
    OUTPUT_NODE = False
    CATEGORY = "MarasIT/upscaling"
    DESCRIPTION = "An \"UPSCALER\" Node"
    FUNCTION = "fn"
    
    @classmethod
    def __get_info(image_width, image_height, image_divisible_by_8, output_image_width, output_image_height):
        return [f"""

    IMAGE (INPUT)
        width   :   {image_width}
        height  :   {image_height}
        image divisible by 8 : {image_divisible_by_8}

    ------------------------------

    IMAGE (OUTPUT)
        width   :   {output_image_width}
        height  :   {output_image_height}
        
    NODE INFO
        version : {VERSION}

"""]        
    
    @classmethod    
    def __get_sigmas(self, sigmas_type, model, steps, denoise, scheduler, model_type):
        if sigmas_type == "SDTurboScheduler":
            SigmaScheduler = getattr(comfy_extras.nodes_custom_sampler, sigmas_type)
            sigmas = SigmaScheduler.get_sigmas(SigmaScheduler, model, steps, denoise)[0]
        elif sigmas_type == "AlignYourStepsScheduler":
            SigmaScheduler = AlignYourStepsScheduler
            sigmas = SigmaScheduler.get_sigmas(SigmaScheduler, model_type, steps, denoise)[0]
        else: # BasicScheduler
            SigmaScheduler = getattr(comfy_extras.nodes_custom_sampler, sigmas_type)
            sigmas = SigmaScheduler.get_sigmas(SigmaScheduler, model, scheduler, steps, denoise)[0]

        return sigmas
    
    @classmethod    
    def fn(self, **kwargs):
        
        # Initialize the bus tuple with None values for each parameter
        image = kwargs.get('image', None)
        upscale_size = kwargs.get('output_size', None)
        upscale_model_name = kwargs.get('upscale_model', None)
        upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader.load_model(comfy_extras.nodes_upscale_model.UpscaleModelLoader, upscale_model_name)[0]
        feather_mask = kwargs.get('feather_mask', None)
        vae = kwargs.get('vae', None)
        tiled = kwargs.get('vae_encode', None)
        tile_size = kwargs.get('tile_size', None)
        model = kwargs.get('model', None)
        noise_seed = seed = kwargs.get('seed', None)
        steps = kwargs.get('steps', None)
        cfg = kwargs.get('cfg', None)
        sampler_name = kwargs.get('sampler_name', None)
        sampler = comfy_extras.nodes_custom_sampler.KSamplerSelect.get_sampler(comfy_extras.nodes_custom_sampler.KSamplerSelect,sampler_name)[0]
        scheduler = kwargs.get('basic_scheduler', None)
        positive = kwargs.get('positive', None)
        negative = kwargs.get('negative', None)
        add_noise = True        
        denoise = kwargs.get('denoise', None)        
        sigmas_type = kwargs.get('sigmas_type', None)
        model_type = kwargs.get('ays_model_type', None)
        sigmas = self.__get_sigmas(sigmas_type, model, steps, denoise, scheduler, model_type)
        max_iterations = kwargs.get('running_count', 1)

        output_info = [f"No info"]
        
        if image is None:
            raise ValueError("MarasitUpscalerRefinerNode id XX: No image provided")

        if not isinstance(image, torch.Tensor):
            raise ValueError("MarasitUpscalerRefinerNode id XX: Image provided is not a Tensor")
        
        log(f"McBoaty is starting to do its magic")
        
        image_width = image.shape[2]
        image_height = image.shape[1]
        image_divisible_by_8 = Image.is_divisible_by_8(image)
        if not image_divisible_by_8:
            image_divisible_by_8 = False
            image_width, image_height = Image.calculate_new_dimensions(image_width, image_height)

        image = nodes.ImageScale.upscale(nodes.ImageScale, image, "nearest-exact", image_width, image_height, "center")[0]
        current_image = image
        for index in range(max_iterations):
            output_image = self.upscale_refine(
                f"{index + 1}/{max_iterations}",
                current_image, 
                upscale_model,
                model, 
                vae, 
                tiled, 
                tile_size, 
                add_noise, 
                noise_seed, 
                cfg, 
                positive, 
                negative, 
                sampler, 
                sigmas, 
                feather_mask
            )
            if not upscale_size: 
                output_image = nodes.ImageScale.upscale(nodes.ImageScale, output_image, "nearest-exact", image_width, image_height, "center")[0]
            current_image = output_image
            
        output_image_width = output_image.shape[2]
        output_image_height = output_image.shape[1]

        output_info = self.__get_info(image_width, image_height, image_divisible_by_8, output_image_width, output_image_height)
        
        log(f"McBoaty is done with its magic")
        
        return (
            output_image,
            image,
            output_info
        )
    
