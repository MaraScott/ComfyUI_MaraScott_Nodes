#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Upscaler Refiner Node.  Upscale and Refine a picture by 2 using a 9 Square Grid to upscale and refine the visual in 9 sequences
#
###

import torch
import math
import comfy
import comfy_extras
import comfy_extras.nodes_custom_sampler
from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
import nodes
import folder_paths

from .McBoaty import UpscalerRefiner_McBoaty

from ...inc.lib.image import Image

from ...utils.log import *

class UpscalerRefiner_McBoaty_v2(UpscalerRefiner_McBoaty):
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {"id":"UNIQUE_ID"},
            "required":{
                "image": ("IMAGE",),
                "output_size": ("BOOLEAN", {"default": True, "label_on": "Upscale size", "label_off": "Input size"}),

                "upscale_model": (folder_paths.get_filename_list("upscale_models"),),

                "feather_mask": ("INT", {"default": 64, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),                
                
                "model": ("MODEL",),
                "vae": ("VAE",),
                "vae_encode": ("BOOLEAN", {"default": True, "label_on": "tiled", "label_off": "standard"}),
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

                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                                

            },
            "optional": {
            }
        }

    RETURN_TYPES = (
        "IMAGE", 
        "IMAGE", 
        "IMAGE",
        "STRING"
    )
    
    RETURN_NAMES = (
        "image", 
        "tiles", 
        "original_resized", 
        "info", 
    )
    
    @classmethod
    def upscale_refine(
        self, 
        iteration,
        image, 
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
    ):
        upscaled_image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel.upscale(comfy_extras.nodes_upscale_model.ImageUpscaleWithModel, upscale_model, image)[0]
        upscaled_width = upscaled_image.shape[2]
        upscaled_height = upscaled_image.shape[1]
        tile_rows = math.floor(upscaled_width // tile_size)
        tile_cols = math.floor(upscaled_height // tile_size)
        
        grid_images = Image.get_grid_images(image, tile_rows, tile_cols)

        grid_upscales = []
        grid_latents = []
        grid_latent_outputs = []
        output_images = []
        total = len(grid_images)
        
        for index, grid_image in enumerate(grid_images):            
            log(f"tile {index + 1}/{total}", None, None, f"Upscaling {iteration}")
            _image_grid = grid_image[:,:,:,:3]
            upscaled_image_grid = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel.upscale(comfy_extras.nodes_upscale_model.ImageUpscaleWithModel, upscale_model, _image_grid)[0]
            grid_upscales.append(upscaled_image_grid)
            
        for index, upscaled_image_grid in enumerate(grid_upscales):
            if tiled == True:
                log(f"tile {index + 1}/{total}", None, None, f"VAEEncodingTiled {iteration}")
                latent_image = nodes.VAEEncodeTiled.encode(nodes.VAEEncodeTiled, vae, upscaled_image_grid, tile_size)[0]
            else:
                log(f"tile {index + 1}/{total}", None, None, f"VAEEncoding {iteration}")
                latent_image = nodes.VAEEncode.encode(nodes.VAEEncode, vae, upscaled_image_grid)[0]
            grid_latents.append(latent_image)
                    
        for index, latent_image in enumerate(grid_latents):            
            log(f"tile {index + 1}/{total}", None, None, f"Refining {iteration}")
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
            if tiled == True:
                log(f"tile {index + 1}/{total}", None, None, f"VAEDecodingTiled {iteration}")
                output = nodes.VAEDecodeTiled.decode(nodes.VAEDecodeTiled, vae, latent_output, tile_size)[0].unsqueeze(0)
            else:
                log(f"tile {index + 1}/{total}", None, None, f"VAEDecoding {iteration}")
                output = nodes.VAEDecode.decode(nodes.VAEDecode, vae, latent_output)[0].unsqueeze(0)
            
            output_images.append(output[0])
            
        output_image, tiles_order = Image.rebuild_image_from_parts(iteration, output_images, upscaled_image, feather_mask, rows = tile_rows, cols = tile_cols)

        tiles_order.sort(key=lambda x: x[0])
        output_tiles = tuple(output for _, output in tiles_order)
        output_tiles = torch.cat(output_tiles)
            
        return output_image, output_tiles
        
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
        sigmas = self._get_sigmas(sigmas_type, model, steps, denoise, scheduler, model_type)
        max_iterations = kwargs.get('running_count', 1)

        output_info = [f"No info"]
        
        if image is None:
            raise ValueError("MaraScottUpscalerRefinerNode id XX: No image provided")

        if not isinstance(image, torch.Tensor):
            raise ValueError("MaraScottUpscalerRefinerNode id XX: Image provided is not a Tensor")
        
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
            output_image, output_tiles = self.upscale_refine(
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

        output_info = self._get_info(image_width, image_height, image_divisible_by_8, output_image_width, output_image_height)
        
        log(f"McBoaty is done with its magic")
        
        return (
            output_image,
            output_tiles,
            image,
            output_info
        )
    
        