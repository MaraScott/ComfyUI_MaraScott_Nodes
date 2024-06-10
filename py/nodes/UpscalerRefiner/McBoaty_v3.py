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
from types import SimpleNamespace
import comfy
import comfy_extras
import comfy_extras.nodes_custom_sampler
from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
import nodes
import folder_paths

from ...utils.version import VERSION
from ...inc.lib.image import MS_Image_v2 as MS_Image
from ...inc.lib.sampler import MS_Sampler

from ...utils.log import *

class UpscalerRefiner_McBoaty_v3():

    SIGMAS_TYPES = [
        "BasicScheduler"
        , "SDTurboScheduler"
        , "AlignYourStepsScheduler"
    ]    
    MODEL_TYPE_SIZES = {
        "SD1": 512,
        "SDXL": 1024,
        "SVD": 1024,
    }
    
    AYS_MODEL_TYPES = list(MODEL_TYPE_SIZES.keys())
    
    INPUTS = {}
    OUTPUTS = {}
    PARAMS = {}
    KSAMPLERS = {}
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "image": ("IMAGE", {"label": "Image" }),
                "output_size": ("BOOLEAN", { "label": "Output Size", "default": True, "label_on": "Upscale size", "label_off": "Input size"}),

                "upscale_model": (folder_paths.get_filename_list("upscale_models"), { "label": "Upscale Model" }),

                "feather_mask": ("INT", { "label": "Feather Mask", "default": 64, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),                
                
                "model": ("MODEL", { "label": "Model" }),
                "vae": ("VAE", { "label": "VAE" }),
                "vae_encode": ("BOOLEAN", { "label": "VAE Encode type", "default": True, "label_on": "tiled", "label_off": "standard"}),
                "positive": ("CONDITIONING", { "label": "Positive" }),
                "negative": ("CONDITIONING", { "label": "Negative" }),

                "tile_size": ("INT", { "label": "Tile Size", "default": 512, "min": 320, "max": 4096, "step": 64}),

                "seed": ("INT", { "label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff}),

                "sigmas_type": (self.SIGMAS_TYPES, { "label": "Sigms Type" }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, { "label": "Sampler Name" }),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, { "label": "Basic Scheduler" }),
                "ays_model_type": (self.AYS_MODEL_TYPES, { "label": "Model Type" }),

                "steps": ("INT", { "label": "Steps", "default": 10, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", { "label": "CFG", "default": 2.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "denoise": ("FLOAT", { "label": "Denoise", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),

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
    
    OUTPUT_NODE = False
    CATEGORY = "MaraScott/upscaling"
    DESCRIPTION = "An \"UPSCALER\" Node"
    FUNCTION = "fn"

    @classmethod    
    def fn(self, **kwargs):
        
        self.init(**kwargs)
        
        if self.INPUTS.image is None:
            raise ValueError("MaraScottUpscalerRefinerNode id XX: No image provided")

        if not isinstance(self.INPUTS.image, torch.Tensor):
            raise ValueError("MaraScottUpscalerRefinerNode id XX: Image provided is not a Tensor")
        
        log(f"McBoaty is starting to do its magic")
        
        self.OUTPUTS.image, image_width, image_height, image_divisible_by_8 = MS_Image().format_2_divby8(self.INPUTS.image)

        current_image = self.OUTPUTS.image
        for index in range(self.PARAMS.max_iterations):
            output_image, output_tiles = self.upscale_refine(current_image, f"{index + 1}/{self.PARAMS.max_iterations}")
            if not self.PARAMS.upscale_size: 
                output_image = nodes.ImageScale.upscale(nodes.ImageScale, output_image, "nearest-exact", image_width, image_height, "center")[0]
            current_image = output_image
            
        output_image_width = output_image.shape[2]
        output_image_height = output_image.shape[1]

        output_info = self._get_info(image_width, image_height, image_divisible_by_8, output_image_width, output_image_height)
        
        log(f"McBoaty is done with its magic")
        
        image = self.OUTPUTS.image
        
        return (
            output_image,
            output_tiles,
            image,
            output_info
        )
        
    @classmethod
    def init(self, **kwargs):
        # Initialize the bus tuple with None values for each parameter
        self.INPUTS = {
            "image": kwargs.get('image', None),
        }
        self.INPUTS = SimpleNamespace(
            image = kwargs.get('image', None),
        )
        self.PARAMS = SimpleNamespace(
            upscale_size = kwargs.get('output_size', None),
            upscale_model_name = kwargs.get('upscale_model', None),
            upscale_method = "lanczos",
            feather_mask = kwargs.get('feather_mask', None),
            max_iterations = kwargs.get('running_count', 1),
        )
        self.PARAMS.upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader().load_model(self.PARAMS.upscale_model_name)[0]

        self.KSAMPLER = SimpleNamespace(
            tiled = kwargs.get('vae_encode', None),
            tile_size = kwargs.get('tile_size', None),
            model = kwargs.get('model', None),
            vae = kwargs.get('vae', None),
            noise_seed = kwargs.get('seed', None),
            sampler_name = kwargs.get('sampler_name', None),
            scheduler = kwargs.get('basic_scheduler', None),
            positive = kwargs.get('positive', None),
            negative = kwargs.get('negative', None),
            add_noise = True,
            sigmas_type = kwargs.get('sigmas_type', None),
            model_type = kwargs.get('ays_model_type', None),
            steps = kwargs.get('steps', None),
            cfg = kwargs.get('cfg', None),
            denoise = kwargs.get('denoise', None),
        )
        self.KSAMPLER.sampler = comfy_extras.nodes_custom_sampler.KSamplerSelect().get_sampler(self.KSAMPLER.sampler_name)[0]
        self.KSAMPLER.tile_size_sampler = self.MODEL_TYPE_SIZES[self.KSAMPLER.model_type]
        self.KSAMPLER.sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, self.KSAMPLER.denoise, self.KSAMPLER.scheduler, self.KSAMPLER.model_type)

        self.OUTPUTS = SimpleNamespace(
            output_info = [f"No info"],        
        )
    
        
    @classmethod
    def _get_info(self, image_width, image_height, image_divisible_by_8, output_image_width, output_image_height):
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
    def _get_sigmas(self, sigmas_type, model, steps, denoise, scheduler, model_type):
        if sigmas_type == "SDTurboScheduler":
            SigmaScheduler = getattr(comfy_extras.nodes_custom_sampler, sigmas_type)
            sigmas = SigmaScheduler().get_sigmas(model, steps, denoise)[0]
        elif sigmas_type == "AlignYourStepsScheduler":
            SigmaScheduler = AlignYourStepsScheduler
            sigmas = SigmaScheduler().get_sigmas(model_type, steps, denoise)[0]
        else: # BasicScheduler
            SigmaScheduler = getattr(comfy_extras.nodes_custom_sampler, sigmas_type)
            sigmas = SigmaScheduler().get_sigmas(model, scheduler, steps, denoise)[0]
        
        return sigmas    
    
    @classmethod
    def upscale_refine(self, image, iteration):
        
        upscaled = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(self.PARAMS.upscale_model, image)[0]
        upscale_coef = upscaled.shape[2] / image.shape[2]
        feather_mask = self.PARAMS.feather_mask
        rows_qty_float = image.shape[2] / self.KSAMPLER.tile_size
        cols_qty_float = image.shape[1] / self.KSAMPLER.tile_size
        rows_qty = math.ceil(rows_qty_float)
        cols_qty = math.ceil(cols_qty_float)
        
        grid_specs = MS_Image().get_dynamic_grid_specs(image.shape[2], image.shape[1], rows_qty, cols_qty, feather_mask)[0]
        
        grid_images = MS_Image().get_grid_images(image, grid_specs)
        
        grid_upscales = []
        grid_latents = []
        grid_latent_outputs = []
        output_images = []
        total = len(grid_images)
        
        for index, grid_image in enumerate(grid_images):            
            log(f"tile {index + 1}/{total}", None, None, f"Upscaling {iteration}")
            # _image_grid = nodes.ImageScaleBy().upscale(_image_grid, self.PARAMS.upscale_method, (_image_grid.shape[2] / self.KSAMPLER.tile_size_sampler))[0]
            upscaled_image_grid = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(self.PARAMS.upscale_model, grid_image)[0]
            grid_upscales.append(upscaled_image_grid)

        for index, upscaled_image_grid in enumerate(grid_upscales):
            
            if self.KSAMPLER.tiled == True:
                log(f"tile {index + 1}/{total}", None, None, f"VAEEncodingTiled {iteration}")
                latent_image = nodes.VAEEncodeTiled().encode(self.KSAMPLER.vae, upscaled_image_grid, self.KSAMPLER.tile_size)[0]
            else:
                log(f"tile {index + 1}/{total}", None, None, f"VAEEncoding {iteration}")
                latent_image = nodes.VAEEncode().encode(self.KSAMPLER.vae, upscaled_image_grid)[0]
            grid_latents.append(latent_image)
                    
        for index, latent_image in enumerate(grid_latents):            
            log(f"tile {index + 1}/{total}", None, None, f"Refining {iteration}")
            latent_output = comfy_extras.nodes_custom_sampler.SamplerCustom().sample(
                self.KSAMPLER.model, 
                self.KSAMPLER.add_noise, 
                self.KSAMPLER.noise_seed, 
                self.KSAMPLER.cfg, 
                self.KSAMPLER.positive, 
                self.KSAMPLER.negative, 
                self.KSAMPLER.sampler, 
                self.KSAMPLER.sigmas, 
                latent_image
            )[0]
            grid_latent_outputs.append(latent_output)

        for index, latent_output in enumerate(grid_latent_outputs):            
            if self.KSAMPLER.tiled == True:
                log(f"tile {index + 1}/{total}", None, None, f"VAEDecodingTiled {iteration}")
                output = (nodes.VAEDecodeTiled().decode(self.KSAMPLER.vae, latent_output, self.KSAMPLER.tile_size)[0].unsqueeze(0))[0]
            else:
                log(f"tile {index + 1}/{total}", None, None, f"VAEDecoding {iteration}")
                output = (nodes.VAEDecode().decode(self.KSAMPLER.vae, latent_output)[0].unsqueeze(0))[0]
            
            # output = nodes.ImageScaleBy().upscale(output, self.PARAMS.upscale_method, (1/(output.shape[2] / self.KSAMPLER.tile_size_sampler)))[0]
            output_images.append(output)

        feather_mask = int(self.PARAMS.feather_mask * upscale_coef)
        upscaled_grid_specs = MS_Image().get_dynamic_grid_specs(upscaled.shape[2], upscaled.shape[1], rows_qty, cols_qty, feather_mask)[0]
        output_image, tiles_order = MS_Image().rebuild_image_from_parts(iteration, output_images, upscaled, upscaled_grid_specs, feather_mask)

        tiles_order.sort(key=lambda x: x[0])
        output_tiles = tuple(output for _, output in tiles_order)            
        output_tiles = torch.cat(output_tiles)

        return output_image, output_tiles
