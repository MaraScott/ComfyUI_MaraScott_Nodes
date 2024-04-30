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

from ..utils.version import VERSION

from ..inc.lib.image import Image

from ..utils.log import *

class UpscalerRefinerNode:
    
    SIGMAS_TYPES = [
        "BasicScheduler"
        , "SDTurboScheduler"
        , "AlignYourStepsScheduler"
    ]
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {"id":"UNIQUE_ID"},
            "required":{
                "image": ("IMAGE",),

                "upscale_model": (folder_paths.get_filename_list("upscale_models"),),
                
                "feather_mask": ("INT", {"default": 350, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),                

                "model": ("MODEL",),
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),

                "seed": ("INT", {"default": 4, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),

                "sigmas_type": (self.SIGMAS_TYPES, ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "ays_model_type": (["SD1", "SDXL", "SVD"], ),

                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),

                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.51, "step": 0.01}),                

            },
            "optional": {
            }
        }

    RETURN_TYPES = (
        "IMAGE", 
        "IMAGE", 
        "STRING"
    )
    
    RETURN_NAMES = (
        "image", 
        "original_resized", 
        "info", 
    )
    
    OUTPUT_NODE = False
    CATEGORY = "MarasIT/upscaling"
    DESCRIPTION = "An \"UPSCALER\" Node"
    FUNCTION = "fn"
    
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
        upscale_model_name = kwargs.get('upscale_model', None)
        upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader.load_model(comfy_extras.nodes_upscale_model.UpscaleModelLoader, upscale_model_name)[0]
        feather_mask = kwargs.get('feather_mask', None)
        vae = kwargs.get('vae', None)
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

        resized_image = nodes.ImageScale.upscale(nodes.ImageScale, image, "nearest-exact", image_width, image_height, "center")[0]

        upscaled_image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel.upscale(comfy_extras.nodes_upscale_model.ImageUpscaleWithModel, upscale_model, resized_image)[0]
        
        grid_images = Image.get_grid_images(resized_image)

        grid_latents = []
        grid_latent_outputs = []
        output_images = []
        total = len(grid_images)
        for index, grid_image in enumerate(grid_images):            
            log(f"Upscaling tile {index + 1}/{total}")
            _image_grid = grid_image[:,:,:,:3]
            upscaled_image_grid = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel.upscale(comfy_extras.nodes_upscale_model.ImageUpscaleWithModel, upscale_model, _image_grid)[0]
            tiled = False
            if tiled == True:
                latent_image = nodes.VAEEncodeTiled.encode(nodes.VAEEncodeTiled, vae, upscaled_image_grid, tile_size)[0]
            else:
                latent_image = nodes.VAEEncode.encode(nodes.VAEEncode, vae, upscaled_image_grid)[0]
            grid_latents.append(latent_image)
                    
        for index, latent_image in enumerate(grid_latents):            
            log(f"Refining tile {index + 1}/{total}")
            latent_output = comfy_extras.nodes_custom_sampler.SamplerCustom.sample(
                comfy_extras.nodes_custom_sampler.SamplerCustom, 
                model, 
                add_noise, 
                noise_seed, 
                cfg, 
                positive, 
                negative, 
                sampler, 
                sigmas, 
                latent_image
            )[0]
            grid_latent_outputs.append(latent_output)

        for index, latent_output in enumerate(grid_latent_outputs):            
            log(f"VAEDecoding tile {index + 1}/{total}")
            if tiled == True:
                output = nodes.VAEDecodeTiled.decode(nodes.VAEDecodeTiled, vae, latent_output, tile_size)[0].unsqueeze(0)
            else:
                output = nodes.VAEDecode.decode(nodes.VAEDecode, vae, latent_output)[0].unsqueeze(0)
            
            output_images.append(output[0])

        output_image = Image.rebuild_image_from_parts(output_images, upscaled_image, feather_mask)
        
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
    
NODE INFO
    version : {VERSION}

"""]
        log(f"McBoaty is done with its magic")
        
        return (
            output_image,
            resized_image,
            output_info
        )
