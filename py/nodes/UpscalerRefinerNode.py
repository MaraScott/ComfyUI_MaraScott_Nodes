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
import nodes
import folder_paths

from ..inc.lib.image import Image

from ..utils.log import *

class UpscalerRefinerNode:
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {"id":"UNIQUE_ID"},
            "required":{
                "image": ("IMAGE",),
                "model_name": (folder_paths.get_filename_list("upscale_models"),),
                
                "feather_mask": ("INT", {"default": 350, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),                

                "model": ("MODEL",),
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),

                "seed": ("INT", {"default": 4, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),

                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                # "latent_image": ("LATENT", ),

                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.51, "step": 0.01}),                

            },
            "optional": {
            }
        }

    RETURN_TYPES = (
        "IMAGE", 
        "IMAGE", 
        "IMAGE", 
        "IMAGE", 
        # "MASK", 
        "STRING"
    )
    
    RETURN_NAMES = (
        "final_image", 
        "refined_image", 
        "upscaled_image", 
        "resized_image", 
        # "mask", 
        "info", 
    )
    
    OUTPUT_NODE = False
    CATEGORY = "MarasIT/upscaling"
    DESCRIPTION = "An \"UPSCALER\" Node"
    FUNCTION = "fn"
    
    def fn(self, **kwargs):
        
        # Initialize the bus tuple with None values for each parameter
        image = kwargs.get('image', None)
        model_name = kwargs.get('model_name', None)
        feather_mask = kwargs.get('feather_mask', None)
        upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader.load_model(comfy_extras.nodes_upscale_model.UpscaleModelLoader, model_name)[0]
        vae = kwargs.get('vae', None)
        tile_size = kwargs.get('tile_size', None)
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

        resized_image = nodes.ImageScale.upscale(nodes.ImageScale, image, "nearest-exact", image_width, image_height, "center")[0]

        upscaled_image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel.upscale(comfy_extras.nodes_upscale_model.ImageUpscaleWithModel, upscale_model, resized_image)[0]
        
        grid_images = Image.get_grid_images(resized_image)

        output_images = []
        for grid_image in grid_images:            
            # Encode the upscaled image using the VAE 
            _image_grid = grid_image[:,:,:,:3]
            upscaled_image_grid = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel.upscale(comfy_extras.nodes_upscale_model.ImageUpscaleWithModel, upscale_model, _image_grid)[0]

            latent_image = nodes.VAEEncodeTiled.encode(nodes.VAEEncodeTiled, vae, upscaled_image_grid, tile_size)[0]
            
            # Use the latent image in the common_ksampler function
            # latent_output = comfy_extras.nodes_custom_sampler.SamplerCustom.sample(
            #     comfy_extras.nodes_custom_sampler.SamplerCustom, 
            #     model, 
            #     add_noise, 
            #     noise_seed, 
            #     cfg, 
            #     positive, 
            #     negative, 
            #     sampler, 
            #     sigmas, 
            #     latent_image
            # )
            latent_output = nodes.KSampler.sample(
                nodes.KSampler,
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
            output = nodes.VAEDecodeTiled.decode(nodes.VAEDecodeTiled, vae, latent_output, tile_size)[0].unsqueeze(0)
            
            # Collect all outputs (you may want to adjust this depending on how you want to handle the outputs)
            output_images.append(output[0])

        refined_image = Image.rebuild_image_from_parts(output_images, upscaled_image, feather_mask)

        latent_image = nodes.VAEEncodeTiled.encode(nodes.VAEEncodeTiled, vae, refined_image, tile_size)[0]

        steps = 10
        cfg = 2.5
        denoise = 0.1
        # Use the latent image in the common_ksampler function
        latent_output = nodes.KSampler.sample(
            nodes.KSampler,
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
        output_image = nodes.VAEDecodeTiled.decode(nodes.VAEDecodeTiled, vae, latent_output, tile_size)[0]
                
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
            refined_image,
            upscaled_image,
            resized_image,
            # output_mask,
            output_info
        )
