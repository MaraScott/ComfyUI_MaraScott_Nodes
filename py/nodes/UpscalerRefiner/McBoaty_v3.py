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
from ...inc.lib.llm import MS_Llm
from ...vendor.ComfyUI_KJNodes.nodes.image_nodes import ColorMatch as ColorMatch

from ...utils.log import *

import time
class UpscalerRefiner_McBoaty_v3():

    UPSCALE_METHODS = [
        "area", 
        "bicubic", 
        "bilinear", 
        "bislerp",
        "lanczos",
        "nearest-exact"
    ]
    
    SIGMAS_TYPES = [
        'BasicScheduler', 
        'SDTurboScheduler', 
        'AlignYourStepsScheduler'
    ]    
    AYS_MODEL_TYPE_SIZES = {
        'SD1': 512,
        'SDXL': 1024,
        'SD3': 1024,
        'SVD': 1024,
    }
    COLOR_MATCH_METHODS = [   
        'none',
        'mkl',
        'hm', 
        'reinhard', 
        'mvgd', 
        'hm-mvgd-hm', 
        'hm-mkl-hm',
    ]
    
    AYS_MODEL_TYPES = list(AYS_MODEL_TYPE_SIZES.keys())
    
    INPUTS = {}
    OUTPUTS = {}
    PARAMS = {}
    KSAMPLERS = {}
    LLM = {}
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "image": ("IMAGE", {"label": "Image" }),

                "model": ("MODEL", { "label": "Model" }),
                "clip": ("CLIP", { "label": "Clip" }),
                "vae": ("VAE", { "label": "VAE" }),
                "positive": ("CONDITIONING", { "label": "Positive" }),
                "negative": ("CONDITIONING", { "label": "Negative" }),
                "seed": ("INT", { "label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff}),

                "upscale_model": (folder_paths.get_filename_list("upscale_models"), { "label": "Upscale Model" }),
                "output_size_type": ("BOOLEAN", { "label": "Output Size Type", "default": True, "label_on": "Upscale size", "label_off": "Custom size"}),
                "output_size": ("FLOAT", { "label": "Custom Output Size", "default": 1.00, "min": 1.00, "max": 16.00, "step":0.01, "round": 0.01}),
                "output_upscale_method": (self.UPSCALE_METHODS, { "label": "Custom Output Upscale Method", "default": "bicubic"}),
                "steps": ("INT", { "label": "Steps", "default": 10, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", { "label": "CFG", "default": 2.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sigmas_type": (self.SIGMAS_TYPES, { "label": "Sigmas Type" }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, { "label": "Sampler Name" }),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, { "label": "Basic Scheduler" }),
                "denoise": ("FLOAT", { "label": "Denoise", "default": 0.27, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ays_model_type": (self.AYS_MODEL_TYPES, { "label": "Model Type", "default": "SDXL" }),
                "tile_size": ("INT", { "label": "Tile Size", "default": 1024, "min": 320, "max": 4096, "step": 64}),
                "vae_encode": ("BOOLEAN", { "label": "VAE Encode type", "default": True, "label_on": "tiled", "label_off": "standard"}),
                "feather_mask": ("INT", { "label": "Feather Mask", "default": 128, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "color_match_method": (self.COLOR_MATCH_METHODS, { "label": "Color Match Method", "default": 'none'}),
                "tile_prompting_active": ("BOOLEAN", { "label": "Tile prompting (experimental)", "default": False, "label_on": "Active", "label_off": "Inactive"}),
                "vision_llm_model": (MS_Llm.VISION_LLM_MODELS, { "label": "Vision LLM Model", "default": "microsoft/kosmos-2-patch14-224" }),
                "llm_model": (MS_Llm.LLM_MODELS, { "label": "LLM Model", "default": "llama3-70b-8192" }),

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

        start_time = time.time()
        
        self.init(**kwargs)
        
        if self.INPUTS.image is None:
            raise ValueError("MaraScottUpscalerRefinerNode id XX: No image provided")

        if not isinstance(self.INPUTS.image, torch.Tensor):
            raise ValueError("MaraScottUpscalerRefinerNode id XX: Image provided is not a Tensor")
        
        log(f"McBoaty is starting to do its magic")
        
        self.OUTPUTS.image, image_width, image_height, image_divisible_by_8 = MS_Image().format_2_divby8(self.INPUTS.image)

        current_image = self.OUTPUTS.image
        for index in range(self.PARAMS.max_iterations):
            output_image, output_tiles, output_prompts = self.upscale_refine(current_image, f"{index + 1}/{self.PARAMS.max_iterations}")
            if not self.PARAMS.upscale_size_type:
                output_image = nodes.ImageScale().upscale(output_image, self.PARAMS.upscale_method, int(image_width * self.PARAMS.upscale_size), int(image_height * self.PARAMS.upscale_size), False)[0]
            current_image = output_image
            
        output_image_width = output_image.shape[2]
        output_image_height = output_image.shape[1]

        end_time = time.time()

        output_info = self._get_info(
            image_width, 
            image_height, 
            image_divisible_by_8, 
            output_image_width, 
            output_image_height,
            output_prompts,
            int(end_time - start_time)
        )
        
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
            upscale_size_type = kwargs.get('output_size_type', None),
            upscale_size = kwargs.get('output_size', None),
            upscale_model_name = kwargs.get('upscale_model', None),
            upscale_method = kwargs.get('output_upscale_method', "lanczos"),
            feather_mask = kwargs.get('feather_mask', None),
            color_match_method = kwargs.get('color_match_method', 'none'),
            max_iterations = kwargs.get('running_count', 1),
            tile_prompting_active = kwargs.get('tile_prompting_active', False),
        )
        self.PARAMS.upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader().load_model(self.PARAMS.upscale_model_name)[0]

        self.KSAMPLER = SimpleNamespace(
            tiled = kwargs.get('vae_encode', None),
            tile_size = kwargs.get('tile_size', None),
            model = kwargs.get('model', None),
            clip = kwargs.get('clip', None),
            vae = kwargs.get('vae', None),
            noise_seed = kwargs.get('seed', None),
            sampler_name = kwargs.get('sampler_name', None),
            scheduler = kwargs.get('basic_scheduler', None),
            positive = kwargs.get('positive', None),
            negative = kwargs.get('negative', None),
            add_noise = True,
            sigmas_type = kwargs.get('sigmas_type', None),
            ays_model_type = kwargs.get('ays_model_type', None),
            steps = kwargs.get('steps', None),
            cfg = kwargs.get('cfg', None),
            denoise = kwargs.get('denoise', None),
        )
        
        self.KSAMPLER.sampler = comfy_extras.nodes_custom_sampler.KSamplerSelect().get_sampler(self.KSAMPLER.sampler_name)[0]
        # isinstance(self.KSAMPLER.model, comfy.model_base.SDXL)
        self.KSAMPLER.tile_size_sampler = self.AYS_MODEL_TYPE_SIZES[self.KSAMPLER.ays_model_type]
        self.KSAMPLER.sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, self.KSAMPLER.denoise, self.KSAMPLER.scheduler, self.KSAMPLER.ays_model_type)
        self.KSAMPLER.outpaint_sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, 1, self.KSAMPLER.scheduler, self.KSAMPLER.ays_model_type)

        self.LLM = SimpleNamespace(
            vision_model = kwargs.get('vision_llm_model', None),
            model = kwargs.get('llm_model', None),
        )

        # TODO : make the feather_mask proportional to tile size ?
        # self.PARAMS.feather_mask = self.KSAMPLER.tile_size // 16

        self.OUTPUTS = SimpleNamespace(
            output_info = [f"No info"],        
        )
    
        
    @classmethod
    def _get_info(self, image_width, image_height, image_divisible_by_8, output_image_width, output_image_height, output_prompts, execution_duration):
        formatted_prompts = "\n".join(f"        [{index+1}] {prompt}" for index, prompt in enumerate(output_prompts))
        
        return [f"""

    IMAGE (INPUT)
        width   :   {image_width}
        height  :   {image_height}
        image divisible by 8 : {image_divisible_by_8}

    ------------------------------

    IMAGE (OUTPUT)
        width   :   {output_image_width}
        height  :   {output_image_height}

    ------------------------------
    
    TILES PROMPTS
{formatted_prompts}    
        
    ------------------------------

    EXECUTION
        DURATION : {execution_duration} seconds

    NODE INFO
        version : {VERSION}

"""]        
    
    @classmethod    
    def _get_sigmas(self, sigmas_type, model, steps, denoise, scheduler, ays_model_type):
        if sigmas_type == "SDTurboScheduler":
            SigmaScheduler = getattr(comfy_extras.nodes_custom_sampler, sigmas_type)
            sigmas = SigmaScheduler().get_sigmas(model, steps, denoise)[0]
        elif sigmas_type == "AlignYourStepsScheduler":
            SigmaScheduler = AlignYourStepsScheduler
            sigmas = SigmaScheduler().get_sigmas(ays_model_type, steps, denoise)[0]
        else: # BasicScheduler
            SigmaScheduler = getattr(comfy_extras.nodes_custom_sampler, sigmas_type)
            sigmas = SigmaScheduler().get_sigmas(model, scheduler, steps, denoise)[0]
        
        return sigmas    
    
    @classmethod
    def upscale_refine(self, image, iteration):
        
        feather_mask = self.PARAMS.feather_mask
        rows_qty_float = image.shape[2] / self.KSAMPLER.tile_size
        cols_qty_float = image.shape[1] / self.KSAMPLER.tile_size
        rows_qty = math.ceil(rows_qty_float)
        cols_qty = math.ceil(cols_qty_float)
        
        grid_specs = MS_Image().get_dynamic_grid_specs(image.shape[2], image.shape[1], rows_qty, cols_qty, feather_mask)[0]
        grid_images = MS_Image().get_grid_images(image, grid_specs)

        grid_prompts = ["No tile prompting"]
        grid_upscales = []
        grid_latents = []
        grid_latent_outputs = []
        output_images = []
        total = len(grid_images)
        
        if self.PARAMS.tile_prompting_active:
            grid_prompts = []
            llm = MS_Llm(self.LLM.vision_model, self.LLM.model)
            prompt_context = llm.vision_llm.generate_prompt(image)

            for index, grid_image in enumerate(grid_images):
                log(f"tile {index + 1}/{total}", None, None, f"Prompting {iteration}")
                prompt_tile = llm.generate_tile_prompt(grid_image, prompt_context, self.KSAMPLER.noise_seed)
                log(prompt_tile, None, None, f"Model {llm.vision_llm.name}")        
                grid_prompts.append(prompt_tile)

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
            positive = self.KSAMPLER.positive
            if self.PARAMS.tile_prompting_active:
                log(f"tile {index + 1}/{total} : {grid_prompts[index]}", None, None, f"ClipTextEncoding {iteration}")
                positive = nodes.CLIPTextEncode().encode(self.KSAMPLER.clip, grid_prompts[index])[0]
            log(f"tile {index + 1}/{total}", None, None, f"Refining {iteration}")
            latent_output = comfy_extras.nodes_custom_sampler.SamplerCustom().sample(
                self.KSAMPLER.model, 
                self.KSAMPLER.add_noise, 
                self.KSAMPLER.noise_seed, 
                self.KSAMPLER.cfg, 
                positive, 
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

        feather_mask = int(self.PARAMS.feather_mask * self.PARAMS.upscale_model.scale)
        upscaled_grid_specs = MS_Image().get_dynamic_grid_specs((image.shape[2]*self.PARAMS.upscale_model.scale), (image.shape[1]*self.PARAMS.upscale_model.scale), rows_qty, cols_qty, feather_mask)[0]
        output_image, tiles_order = MS_Image().rebuild_image_from_parts(iteration, output_images, image, upscaled_grid_specs, feather_mask, self.PARAMS.upscale_model.scale, grid_prompts)

        if self.PARAMS.color_match_method != 'none':
            output_image = ColorMatch().colormatch(image, output_image, self.PARAMS.color_match_method)[0]

        tiles_order.sort(key=lambda x: x[0])
        output_tiles = tuple(output for _, output, _ in tiles_order)
        output_tiles = torch.cat(output_tiles)
        output_prompts = tuple(prompt for _, _, prompt in tiles_order)

        return output_image, output_tiles, output_prompts
