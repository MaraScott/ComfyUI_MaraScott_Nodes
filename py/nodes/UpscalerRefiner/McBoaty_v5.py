#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Upscaler Refiner Node.  Upscale and Refine a picture by 2 using a 9 Square Grid to upscale and refine the visual in 9 sequences
#
###

import os
import time
import glob
import torch
import math
from types import SimpleNamespace
import comfy
import comfy_extras
import comfy_extras.nodes_custom_sampler
from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
from comfy_extras.nodes_canny import Canny
import nodes
from server import PromptServer
from aiohttp import web
import folder_paths

from PIL import Image
import numpy as np

from .... import root_dir, __MARASCOTT_TEMP__
from ...utils.version import VERSION
from ...inc.lib.image import MS_Image_v2 as MS_Image
from ...vendor.ComfyUI_KJNodes.nodes.image_nodes import ColorMatch as ColorMatch
from ...inc.lib.llm import MS_Llm
from ...inc.lib.cache import MS_Cache

from .inc.prompt import Node as NodePrompt

from ...utils.log import log, get_log, COLORS


class McBoaty_Upscaler_v5():

    UPSCALE_METHODS = [
        "area", 
        "bicubic", 
        "bilinear", 
        "bislerp",
        "lanczos",
        "nearest-exact"
    ]
    
    COLOR_MATCH_METHODS = [   
        'none',
        'mkl',
        'hm', 
        'reinhard', 
        'mvgd', 
        'hm-mvgd-hm', 
        'hm-mkl-hm',
    ]
    
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
                "output_upscale_method": (self.UPSCALE_METHODS, { "label": "Custom Output Upscale Method", "default": "bicubic"}),

                "tile_size": ("INT", { "label": "Tile Size", "default": 512, "min": 320, "max": 4096, "step": 64}),
                "feather_mask": ("INT", { "label": "Feather Mask", "default": 64, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "vae_encode": ("BOOLEAN", { "label": "VAE Encode type", "default": True, "label_on": "tiled", "label_off": "standard"}),
                "tile_size_vae": ("INT", { "label": "Tile Size (VAE)", "default": 512, "min": 320, "max": 4096, "step": 64}),

                "color_match_method": (self.COLOR_MATCH_METHODS, { "label": "Color Match Method", "default": 'none'}),
                "tile_prompting_active": ("BOOLEAN", { "label": "Tile prompting (with WD14 Tagger - experimental)", "default": False, "label_on": "Active", "label_off": "Inactive"}),
                "vision_llm_model": (MS_Llm.VISION_LLM_MODELS, { "label": "Vision LLM Model", "default": "microsoft/Florence-2-large" }),
                "llm_model": (MS_Llm.LLM_MODELS, { "label": "LLM Model", "default": "llama3-70b-8192" }),

            },
            "optional": {
            }
        }

    RETURN_TYPES = (
        "MC_BOATY_PIPE", 
        "MC_PROMPTY_PIPE_IN",
        "STRING",
    )
    
    RETURN_NAMES = (
        "McBoaty Pipe",
        "McPrompty Pipe",
        "info", 
    )
    
    OUTPUT_IS_LIST = (
        False,
        False,
        False,
    )
    
    
    OUTPUT_NODE = True
    CATEGORY = "MaraScott/upscaling"
    DESCRIPTION = "An \"UPSCALER\" Node"
    FUNCTION = "fn"

    @classmethod    
    def fn(self, **kwargs):

        start_time = time.time()
        
        self.init(**kwargs)
        
        if self.INPUTS.image is None:
            raise ValueError(f"MaraScottUpscalerRefinerNode id {self.INFO.id}: No image provided")

        if not isinstance(self.INPUTS.image, torch.Tensor):
            raise ValueError(f"MaraScottUpscalerRefinerNode id {self.INFO.id}: Image provided is not a Tensor")
        
        log("McBoaty (Upscaler) is starting to do its magic", None, None, f"Node {self.INFO.id}")
        
        self.OUTPUTS.image, image_width, image_height, image_divisible_by_8 = MS_Image().format_2_divby8(self.INPUTS.image)

        self.PARAMS.grid_specs, self.OUTPUTS.grid_images, self.OUTPUTS.grid_prompts = self.upscale(self.OUTPUTS.image, "Upscaling")

        end_time = time.time()

        output_info = self._get_info(
            image_width, 
            image_height, 
            image_divisible_by_8, 
            self.OUTPUTS.grid_prompts,
            int(end_time - start_time)
        )
        
        log("McBoaty (Upscaler) is done with its magic", None, None, f"Node {self.INFO.id}")

        output_tiles = torch.cat(self.OUTPUTS.grid_images)

        return (
            (
                self.INPUTS,
                self.PARAMS,
                self.KSAMPLER,
                self.OUTPUTS,
            ),
            (
                self.OUTPUTS.grid_prompts,
                output_tiles,
            ),
            output_info
        )
        
    @classmethod
    def init(self, **kwargs):
        # Initialize the bus tuple with None values for each parameter
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', None),
        )
        
        self.INPUTS = SimpleNamespace(
            image = kwargs.get('image', None),
        )
        
        self.LLM = SimpleNamespace(
            vision_model = kwargs.get('vision_llm_model', None),
            model = kwargs.get('llm_model', None),
        )
        
        self.PARAMS = SimpleNamespace(
            upscale_model_name = kwargs.get('upscale_model', None),
            upscale_method = kwargs.get('output_upscale_method', "lanczos"),
            feather_mask = kwargs.get('feather_mask', None),
            color_match_method = kwargs.get('color_match_method', 'none'),
            upscale_size_type = None,
            upscale_size = None,
            tile_prompting_active = kwargs.get('tile_prompting_active', False),
            grid_spec = None,
            rows_qty = 1,
            cols_qty = 1,
        )
        self.PARAMS.upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader().load_model(self.PARAMS.upscale_model_name)[0]

        self.KSAMPLER = SimpleNamespace(
            tiled = kwargs.get('vae_encode', None),
            tile_size = kwargs.get('tile_size', None),
            tile_size_vae = kwargs.get('tile_size_vae', None),
            model = kwargs.get('model', None),
            clip = kwargs.get('clip', None),
            vae = kwargs.get('vae', None),
            noise_seed = kwargs.get('seed', None),
            sampler_name = None,
            scheduler = None,
            positive = kwargs.get('positive', None),
            negative = kwargs.get('negative', None),
            add_noise = True,
            sigmas_type = None,
            model_type = None,
            steps = None,
            cfg = None,
            denoise = None,
            control_net_name = None,
            control = None,
        )

        # TODO : make the feather_mask proportional to tile size ?
        # self.PARAMS.feather_mask = self.KSAMPLER.tile_size // 16

        self.OUTPUTS = SimpleNamespace(
            grid_images = [],
            grid_prompts = [],
            output_info = ["No info"],
            grid_tiles_to_process = [],
        )
    
        
    @classmethod
    def _get_info(self, image_width, image_height, image_divisible_by_8, output_prompts, execution_duration):
        formatted_prompts = "\n".join(f"        [{index+1}] {prompt}" for index, prompt in enumerate(output_prompts))
        
        return [f"""

    IMAGE (INPUT)
        width   :   {image_width}
        height  :   {image_height}
        image divisible by 8 : {image_divisible_by_8}

    ------------------------------

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
    def upscale(self, image, iteration):
        
        feather_mask = self.PARAMS.feather_mask
        rows_qty_float = (image.shape[1] * self.PARAMS.upscale_model.scale) / self.KSAMPLER.tile_size
        cols_qty_float = (image.shape[2] * self.PARAMS.upscale_model.scale) / self.KSAMPLER.tile_size
        rows_qty = math.ceil(rows_qty_float)
        cols_qty = math.ceil(cols_qty_float)

        tiles_qty = rows_qty * cols_qty        
        if tiles_qty > 64 :
            msg = get_log(f"\n\n--------------------\n\n!!! Number of tiles is higher than 64 ({tiles_qty} for {self.PARAMS.cols_qty} cols and {self.PARAMS.rows_qty} rows)!!!\n\nPlease consider increasing your tile and feather sizes\n\n--------------------\n", "BLUE", "YELLOW", f"Node {self.INFO.id} - McBoaty_Upscaler_v5")
            raise ValueError(msg)

        upscaled_image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(self.PARAMS.upscale_model, image)[0]

        self.PARAMS.rows_qty = rows_qty
        self.PARAMS.cols_qty = cols_qty
        
        
        # grid_specs = MS_Image().get_dynamic_grid_specs(upscaled_image.shape[2], upscaled_image.shape[1], rows_qty, cols_qty, feather_mask)[0]
        grid_specs = MS_Image().get_tiled_grid_specs(upscaled_image, self.KSAMPLER.tile_size, self.PARAMS.rows_qty, self.PARAMS.cols_qty, feather_mask)[0]
        grid_images = MS_Image().get_grid_images(upscaled_image, grid_specs)
        
        grid_prompts = []
        llm = MS_Llm(self.LLM.vision_model, self.LLM.model)
        prompt_context = llm.vision_llm.generate_prompt(image)
        total = len(grid_images)
        for index, grid_image in enumerate(grid_images):
            prompt_tile = prompt_context
            if self.PARAMS.tile_prompting_active:
                log(f"tile {index + 1}/{total} - [tile prompt]", None, None, f"Node {self.INFO.id} - Prompting {iteration}")
                prompt_tile = llm.generate_tile_prompt(grid_image, prompt_context, self.KSAMPLER.noise_seed)
            log(f"tile {index + 1}/{total} - [tile prompt] {prompt_tile}", None, None, f"Node {self.INFO.id} - Prompting {iteration}")
            grid_prompts.append(prompt_tile)
                            
        return grid_specs, grid_images, grid_prompts

class McBoaty_Refiner_v5():

    SIGMAS_TYPES = [
        'BasicScheduler', 
        'SDTurboScheduler', 
        'AlignYourStepsScheduler'
    ]    
    MODEL_TYPE_SIZES = {
        'SD1': 512,
        'SDXL': 1024,
        'SD3': 1024,
        'FLUX1': 1024,
        'SVD': 1024,
    }
    MODEL_TYPES = list(MODEL_TYPE_SIZES.keys())

    CONTROLNETS = folder_paths.get_filename_list("controlnet")
    CONTROLNET_CANNY_ONLY = ["None"]+[controlnet_name for controlnet_name in CONTROLNETS if controlnet_name is not None and ('canny' in controlnet_name.lower() or 'union' in controlnet_name.lower())]
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "pipe": ("MC_BOATY_PIPE", {"label": "McBoaty Pipe" }),
                "tiles_to_process": ("STRING", { "label": "Tile to process", "default": ''}),
                "output_size_type": ("BOOLEAN", { "label": "Output Size Type", "default": True, "label_on": "Upscale size", "label_off": "Custom size"}),
                "output_size": ("FLOAT", { "label": "Custom Output Size", "default": 1.00, "min": 1.00, "max": 16.00, "step":0.01, "round": 0.01}),
                "sigmas_type": (self.SIGMAS_TYPES, { "label": "Sigmas Type" }),
                "model_type": (self.MODEL_TYPES, { "label": "Model Type", "default": "SDXL" }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, { "label": "Sampler Name" }),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, { "label": "Basic Scheduler" }),
                "steps": ("INT", { "label": "Steps", "default": 10, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", { "label": "CFG", "default": 2.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "denoise": ("FLOAT", { "label": "Denoise", "default": 0.27, "min": 0.0, "max": 1.0, "step": 0.01}),
                "control_net_name": (self.CONTROLNET_CANNY_ONLY , { "label": "ControlNet (Canny only)", "default": "None" }),
                "low_threshold": ("FLOAT", {"label": "Low Threshold (Canny)", "default": 0.6, "min": 0.01, "max": 0.99, "step": 0.01}),
                "high_threshold": ("FLOAT", {"label": "High Threshold (Canny)", "default": 0.6, "min": 0.01, "max": 0.99, "step": 0.01}),
                "strength": ("FLOAT", {"label": "Strength (ControlNet)", "default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"label": "Start % (ControlNet)", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"label": "End % (ControlNet)", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                "pipe_prompty": ("MC_PROMPTY_PIPE_OUT", {"label": "McPrompty Pipe" }),
            }
        }

    RETURN_TYPES = (
        "MC_BOATY_PIPE", 
        "MC_PROMPTY_PIPE_IN", 
        "IMAGE", 
        "IMAGE", 
        "IMAGE",
        "STRING",
        "STRING"
    )
    
    RETURN_NAMES = (
        "McBoaty Pipe", 
        "McPrompty Pipe", 
        "image", 
        "image (original)", 
        "tiles", 
        "prompts", 
        "info", 
    )
    
    OUTPUT_IS_LIST = (False,) * len(RETURN_TYPES)
    
    
    OUTPUT_NODE = True
    CATEGORY = "MaraScott/upscaling"
    DESCRIPTION = "A \"Refiner\" Node"
    FUNCTION = "fn"

    @classmethod    
    def fn(self, **kwargs):
        
        start_time = time.time()
        
        self.init(**kwargs)

        log("McBoaty (Refiner) is starting to do its magic", None, None, f"Node {self.INFO.id}")
        
        INPUTS = self.INPUTS
        PARAMS = self.PARAMS
        KSAMPLER = self.KSAMPLER
        OUTPUTS = self.OUTPUTS
        
        PARAMS.grid_prompts, OUTPUTS.output_image, OUTPUTS.output_tiles, OUTPUTS.grid_tiles_to_process = self.refine(self.OUTPUTS.image, "Upscaling")
        
        end_time = time.time()

        output_info = self._get_info(
            int(end_time - start_time)
        )
        
        output_tiles = torch.cat(self.OUTPUTS.grid_images)

        log("McBoaty (Refiner) is done with its magic", None, None, f"Node {self.INFO.id}")
        
        return (
            (
                INPUTS,
                PARAMS,
                KSAMPLER,
                OUTPUTS,
            ),
            (
                self.OUTPUTS.grid_prompts,
                output_tiles,
            ),            
            OUTPUTS.output_image, 
            OUTPUTS.image, 
            OUTPUTS.output_tiles, 
            PARAMS.grid_prompts, 
            output_info, 
        )
        
    @classmethod
    def init(self, **kwargs):
        
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', None),
        )
        
        attribute_names = ('INPUTS', 'PARAMS', 'KSAMPLER', 'OUTPUTS') 
        pipe = kwargs.get('pipe', (None,) * len(attribute_names))

        for name, value in zip(attribute_names, pipe):
            setattr(self, name, value)

        self.PARAMS.upscale_size_type = kwargs.get('output_size_type', None)
        self.PARAMS.upscale_size = kwargs.get('output_size', None)
        _tiles_to_process = kwargs.get('tiles_to_process', '')
        self.PARAMS.tiles_to_process = self.set_tiles_to_process(_tiles_to_process)
        
        self.KSAMPLER.sampler_name = kwargs.get('sampler_name', None)
        self.KSAMPLER.scheduler = kwargs.get('basic_scheduler', None)
        self.KSAMPLER.sigmas_type = kwargs.get('sigmas_type', None)
        self.KSAMPLER.model_type = kwargs.get('model_type', None)
        self.KSAMPLER.steps = kwargs.get('steps', None)
        self.KSAMPLER.cfg = kwargs.get('cfg', None)
        self.KSAMPLER.denoise = kwargs.get('denoise', None)
                
        self.KSAMPLER.sampler = comfy_extras.nodes_custom_sampler.KSamplerSelect().get_sampler(self.KSAMPLER.sampler_name)[0]
        self.KSAMPLER.tile_size_sampler = self.MODEL_TYPE_SIZES[self.KSAMPLER.model_type]
        self.KSAMPLER.sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, self.KSAMPLER.denoise, self.KSAMPLER.scheduler, self.KSAMPLER.model_type)
        self.KSAMPLER.outpaint_sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, 1, self.KSAMPLER.scheduler, self.KSAMPLER.model_type)

        self.CONTROLNET = SimpleNamespace(
            name = kwargs.get('control_net_name', 'None'),
            path = None,
            controlnet = None,
            low_threshold = kwargs.get('low_threshold', None),
            high_threshold = kwargs.get('high_threshold', None),
            strength = kwargs.get('strength', None),
            start_percent = kwargs.get('start_percent', None),
            end_percent = kwargs.get('end_percent', None),
        )
        if self.CONTROLNET.name != "None":
            self.CONTROLNET.path = folder_paths.get_full_path("controlnet", self.CONTROLNET.name)
            self.CONTROLNET.controlnet = comfy.controlnet.load_controlnet(self.CONTROLNET.path)
            
        grid_prompts, grid_denoises = kwargs.get('pipe_prompty', (None, None))
        
        if grid_prompts is None:
            grid_prompts = (None,) * len(self.OUTPUTS.grid_prompts)
            grid_denoises = (None,) * len(self.OUTPUTS.grid_prompts)
            
        grid_prompts = list([gp if gp is not None else default_gp for gp, default_gp in zip(grid_prompts, self.OUTPUTS.grid_prompts)])
        for i, prompt in enumerate(grid_prompts):
            if prompt is None:
                grid_prompts[i] = self.OUTPUTS.grid_prompts[i]
        self.OUTPUTS.grid_prompts = grid_prompts

        grid_denoises = list([gp if gp is not None else default_gp for gp, default_gp in zip(grid_denoises, ("",) * len(self.OUTPUTS.grid_prompts))])
        for i, denoise in enumerate(grid_denoises):
            if denoise == "":
                grid_denoises[i] = self.KSAMPLER.denoise
            else:
                grid_denoises[i] = float(grid_denoises[i])
        self.OUTPUTS.grid_denoises = grid_denoises
        
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
    def set_tiles_to_process(self, tiles_to_process=''):

        max_tiles = len(self.OUTPUTS.grid_tiles_to_process)
        max = max_tiles if max_tiles > 0 else 64
        
        def is_valid_index(index, max = 64):
            return 1 <= index <= max
        def to_computer_index(human_index):
            return human_index - 1

        _tiles_to_process = []
        
        if tiles_to_process == '':
            return _tiles_to_process

        indexes = tiles_to_process.split(',')
        
        for index in indexes:
            index = index.strip()
            if '-' in index:
                # Range of indexes
                start, end = map(int, index.split('-'))
                if is_valid_index(start, max) and is_valid_index(end, max):
                    _tiles_to_process.extend(range(to_computer_index(start), to_computer_index(end) + 1))
                else:
                    _tiles_to_process.append(-1)
                    log(f"tiles_to_process is not in valid format '{tiles_to_process}' - Allowed formats : indexes from 1 to {max} or any range like 1-{max}", None, COLORS['YELLOW'], f"Node {self.INFO.id}")
            else:
                # Single index
                try:
                    index = int(index)
                    if is_valid_index(index, max):
                        _tiles_to_process.append(to_computer_index(index))
                    else:
                        _tiles_to_process.append(-1)
                        log(f"tiles_to_process is not in valid format '{tiles_to_process}' - Allowed formats : indexes from 1 to {max} or any range like 1-{max}", None, COLORS['YELLOW'], f"Node {self.INFO.id}")
                except ValueError:
                    _tiles_to_process.append(-1)
                    # Ignore non-integer values
                    pass

        # Remove duplicates and sort
        _tiles_to_process = sorted(set(_tiles_to_process))
        if -1 in _tiles_to_process:
            _tiles_to_process = [-1]

        return _tiles_to_process
            
    @classmethod
    def _get_info(self, execution_duration):
        
        return [f"""

    EXECUTION
        DURATION : {execution_duration} seconds

    NODE INFO
        version : {VERSION}

"""]        
        
    @classmethod
    def refine(self, image, iteration):
        
        grid_latents = []
        grid_latent_outputs = []
        output_images = []
        total = len(self.OUTPUTS.grid_images)
        
        for index, upscaled_image_grid in enumerate(self.OUTPUTS.grid_images):
            latent_image = None
            if len(self.PARAMS.tiles_to_process) == 0 or index in self.PARAMS.tiles_to_process:
                if self.KSAMPLER.tiled:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEEncodingTiled {iteration}")
                    latent_image = nodes.VAEEncodeTiled().encode(self.KSAMPLER.vae, upscaled_image_grid, self.KSAMPLER.tile_size_vae)[0]
                else:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEEncoding {iteration}")
                    latent_image = nodes.VAEEncode().encode(self.KSAMPLER.vae, upscaled_image_grid)[0]
            grid_latents.append(latent_image)
        
        for index, latent_image in enumerate(grid_latents):
            latent_output = None
            if len(self.PARAMS.tiles_to_process) == 0 or index in self.PARAMS.tiles_to_process:

                positive = self.KSAMPLER.positive
                negative = self.KSAMPLER.negative

                sigmas = self.KSAMPLER.sigmas
                if self.OUTPUTS.grid_denoises[index] != self.KSAMPLER.denoise:
                    denoise = self.OUTPUTS.grid_denoises[index]
                    sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, self.OUTPUTS.grid_denoises[index], self.KSAMPLER.scheduler, self.KSAMPLER.model_type)
                else:
                    denoise = self.KSAMPLER.denoise
                    
                log(f"tile {index + 1}/{total} : {denoise} / {self.OUTPUTS.grid_prompts[index]}", None, None, f"Node {self.INFO.id} - Denoise/ClipTextEncoding {iteration}")
                positive = nodes.CLIPTextEncode().encode(self.KSAMPLER.clip, self.OUTPUTS.grid_prompts[index])[0]
                if self.CONTROLNET.controlnet is not None:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - Canny {iteration}")
                    canny_image = Canny().detect_edge(self.OUTPUTS.grid_images[index], self.CONTROLNET.low_threshold, self.CONTROLNET.high_threshold)[0]
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - ControlNetApply {iteration}")
                    positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive, negative, self.CONTROLNET.controlnet, canny_image, self.CONTROLNET.strength, self.CONTROLNET.start_percent, self.CONTROLNET.end_percent, self.KSAMPLER.vae )
                    
                log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - Refining {iteration}")
                latent_output = comfy_extras.nodes_custom_sampler.SamplerCustom().sample(
                    self.KSAMPLER.model, 
                    self.KSAMPLER.add_noise, 
                    self.KSAMPLER.noise_seed, 
                    self.KSAMPLER.cfg, 
                    positive, 
                    negative, 
                    self.KSAMPLER.sampler, 
                    sigmas, 
                    latent_image
                )[0]
            grid_latent_outputs.append(latent_output)

        for index, latent_output in enumerate(grid_latent_outputs):            
            output = None
            if len(self.PARAMS.tiles_to_process) == 0 or index in self.PARAMS.tiles_to_process:
                if self.KSAMPLER.tiled:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEDecodingTiled {iteration}")
                    output = (nodes.VAEDecodeTiled().decode(self.KSAMPLER.vae, latent_output, self.KSAMPLER.tile_size_vae)[0].unsqueeze(0))[0]
                else:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEDecoding {iteration}")
                    output = (nodes.VAEDecode().decode(self.KSAMPLER.vae, latent_output)[0].unsqueeze(0))[0]            
            output_images.append(output)

        if len(self.PARAMS.tiles_to_process) > 0 and len(self.OUTPUTS.grid_tiles_to_process) == 0:
            log("!!! WARNING !!! you are processing specific tiles without a fully refined image, we suggest to pass through a full refiner first", None, COLORS['YELLOW'], f"Node {self.INFO.id} {iteration}")
            self.OUTPUTS.grid_tiles_to_process = self.OUTPUTS.grid_images

        if len(self.PARAMS.tiles_to_process) > 0:
            _grid_tiles_to_process = list(self.OUTPUTS.grid_tiles_to_process)
            for index, output_image in enumerate(output_images):
                if output_image is None:
                    output_images[index] = _grid_tiles_to_process[index]
                    
        output_images = tuple(output_images)

        feather_mask = self.PARAMS.feather_mask
        output_image, tiles_order = MS_Image().rebuild_image_from_parts(iteration, output_images, image, self.PARAMS.grid_specs, feather_mask, self.PARAMS.upscale_model.scale, self.PARAMS.rows_qty, self.PARAMS.cols_qty, self.OUTPUTS.grid_prompts)

        if self.PARAMS.color_match_method != 'none':
            output_image = ColorMatch().colormatch(image, output_image, self.PARAMS.color_match_method)[0]

        if not self.PARAMS.upscale_size_type:
            output_image = nodes.ImageScale().upscale(output_image, self.PARAMS.upscale_method, int(self.OUTPUTS.image.shape[2] * self.PARAMS.upscale_size), int(self.OUTPUTS.image.shape[1] * self.PARAMS.upscale_size), False)[0]

        _tiles_order = tuple(output for _, output, _ in tiles_order)
        tiles_order.sort(key=lambda x: x[0])
        output_tiles = tuple(output for _, output, _ in tiles_order)
        output_tiles = torch.cat(output_tiles)
        output_prompts = tuple(prompt for _, _, prompt in tiles_order)

        return output_prompts, output_image, output_tiles, _tiles_order

class McBoaty_TilePrompter_v5():

    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{    
                "pipe": ("MC_PROMPTY_PIPE_IN", {"label": "McPrompty Pipe" }),
            },
            "optional": {
                "requeue": ("INT", { "label": "requeue (automatic or manual)", "default": 0, "min": 0, "max": 99999999999, "step": 1}),                
                **NodePrompt.ENTRIES,
            }
        }

    RETURN_TYPES = (
        "MC_PROMPTY_PIPE_OUT",
    )
    
    RETURN_NAMES = (
        "McPrompty Pipe",
    )
    
    OUTPUT_IS_LIST = (
        False,
    )
        
    OUTPUT_NODE = True
    CATEGORY = "MaraScott/upscaling"
    DESCRIPTION = "A \"Tile Prompt Editor\" Node"
    FUNCTION = "fn"

    @classmethod    
    def fn(self, **kwargs):
                        
        input_prompts, input_tiles = kwargs.get('pipe', (None, None))
        input_denoises = ('', ) * len(input_prompts)

        self.init(**kwargs)
        
        log("McBoaty (PromptEditor) is starting to do its magic", None, None, f"Node {self.INFO.id}")
        
        _input_prompts = MS_Cache.get(self.CACHE.prompt, input_prompts)
        _input_prompts_edited = MS_Cache.get(self.CACHE.prompt_edited, input_prompts)
        _input_denoises = MS_Cache.get(self.CACHE.denoise, input_denoises)
        _input_denoises_edited = MS_Cache.get(self.CACHE.denoise_edited, input_denoises)
        
        refresh = False
        
        if not MS_Cache.isset(self.CACHE.denoise):
            _input_denoises = input_denoises
            MS_Cache.set(self.CACHE.denoise, _input_denoises)
        if not MS_Cache.isset(self.CACHE.prompt) or _input_prompts != input_prompts:
            _input_prompts = input_prompts
            MS_Cache.set(self.CACHE.prompt, _input_prompts)
            _input_denoises = input_denoises
            MS_Cache.set(self.CACHE.denoise, input_denoises)
            refresh = True

        if not MS_Cache.isset(self.CACHE.denoise_edited) or refresh:
            _input_denoises_edited = input_denoises
            MS_Cache.set(self.CACHE.denoise_edited, _input_denoises_edited)
        if not MS_Cache.isset(self.CACHE.prompt_edited) or refresh:
            _input_prompts_edited = input_prompts
            MS_Cache.set(self.CACHE.prompt_edited, _input_prompts_edited)
            _input_denoises_edited = input_denoises
            MS_Cache.set(self.CACHE.denoise_edited, _input_denoises_edited)
        elif len(_input_prompts_edited) != len(_input_prompts):
            _input_prompts_edited = [gp if gp is not None else default_gp for gp, default_gp in zip(_input_prompts_edited, input_prompts)]
            MS_Cache.set(self.CACHE.prompt_edited, _input_prompts_edited)
            _input_denoises_edited = [gp if gp is not None else default_gp for gp, default_gp in zip(_input_denoises_edited, input_denoises)]
            MS_Cache.set(self.CACHE.denoise_edited, _input_denoises_edited)

        if _input_denoises_edited != _input_denoises:
            input_denoises = _input_denoises_edited
        if _input_prompts_edited != _input_prompts:
            input_prompts = _input_prompts_edited

        output_prompts_js = input_prompts
        input_prompts_js = _input_prompts
        output_prompts = output_prompts_js
        output_denoises_js = input_denoises
        input_denoises_js = _input_denoises
        output_denoises = output_denoises_js

        results = list()
        filename_prefix = "McBoaty" + "_temp_" + "tilePrompter" + "_id_" + self.INFO.id
        search_pattern = os.path.join(__MARASCOTT_TEMP__, filename_prefix + '*')
        files_to_delete = glob.glob(search_pattern)
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                # log(f"Deleted: {file_path}", None, None, f"Node {self.INFO.id} - SUCCESS")
            except Exception as e:
                log(f"Error deleting {file_path}: {e}", None, None, "Node {self.INFO.id} - ERROR")        
            
        for index, tile in enumerate(input_tiles):
            full_output_folder, filename, counter, subfolder, subfolder_filename_prefix = folder_paths.get_save_image_path(f"MaraScott/{filename_prefix}", self.output_dir, tile.shape[1], tile.shape[0])
            file = f"{filename}_{index:05}.png"
            file_path = os.path.join(full_output_folder, file)
            
            if not os.path.exists(file_path):
                i = 255. * tile.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)                

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "temp"
            })
            counter += 1

        log("McBoaty (PromptEditor) is done with its magic", None, None, f"Node {self.INFO.id}")
                    
        return {"ui": {
            "prompts_out": output_prompts_js, 
            "prompts_in": input_prompts_js , 
            "denoises_out": output_denoises_js, 
            "denoises_in": input_denoises_js , 
            "tiles": results,
        }, "result": ((output_prompts, output_denoises),)}

    @classmethod
    def init(self, **kwargs):
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', 0),
        )
        self.CACHE = SimpleNamespace(
            prompt = f'input_prompts_{self.INFO.id}',
            prompt_edited = None,
            denoise = f'input_denoises_{self.INFO.id}',
            denoise_edited = None,
        )
        self.CACHE.prompt_edited = f'{self.CACHE.prompt}_edited'
        self.CACHE.denoise_edited = f'{self.CACHE.denoise}_edited'
        
        self.output_dir = folder_paths.get_temp_directory()
        
@PromptServer.instance.routes.get("/MaraScott/McBoaty/v5/get_input_prompts")
async def get_input_prompts(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_prompts_{nodeId}'
    input_prompts = MS_Cache.get(cache_name, [])
    return web.json_response({ "prompts_in": input_prompts })
    
@PromptServer.instance.routes.get("/MaraScott/McBoaty/v5/get_input_denoises")
async def get_input_denoises(request):
    nodeId = request.query.get("node", None)
    cache_name = f'input_denoises_{nodeId}'
    input_denoises = MS_Cache.get(cache_name, [])
    return web.json_response({ "denoises_in": input_denoises })
    
@PromptServer.instance.routes.get("/MaraScott/McBoaty/v5/set_prompt")
async def set_prompt(request):
    prompt = request.query.get("prompt", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    # clientId = request.query.get("clientId", None)
    cache_name = f'input_prompts_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _input_prompts = MS_Cache.get(cache_name, [])
    _input_prompts_edited = MS_Cache.get(cache_name_edited, _input_prompts)
    if _input_prompts_edited and index < len(_input_prompts_edited):
        _input_prompts_edited_list = list(_input_prompts_edited)
        _input_prompts_edited_list[index] = prompt
        _input_prompts_edited = tuple(_input_prompts_edited_list)
        MS_Cache.set(cache_name_edited, _input_prompts_edited)
    return web.json_response(f"Tile {index} prompt has been updated :{prompt}")

@PromptServer.instance.routes.get("/MaraScott/McBoaty/v5/set_denoise")
async def set_denoise(request):
    denoise = request.query.get("denoise", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    # clientId = request.query.get("clientId", None)
    cache_name = f'input_denoises_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _input_denoises = MS_Cache.get(cache_name, [])
    _input_denoises_edited = MS_Cache.get(cache_name_edited, _input_denoises)
    if _input_denoises_edited and index < len(_input_denoises_edited):
        _input_denoises_edited_list = list(_input_denoises_edited)
        _input_denoises_edited_list[index] = denoise
        _input_denoises_edited = tuple(_input_denoises_edited_list)
        MS_Cache.set(cache_name_edited, _input_denoises_edited)
    return web.json_response(f"Tile {index} denoise has been updated: {denoise}")

@PromptServer.instance.routes.get("/MaraScott/McBoaty/v5/tile_prompt")
async def tile_prompt(request):
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)

    type = request.query.get("type", "output")
    if type not in ["output", "input", "temp"]:
        return web.Response(status=400)

    target_dir = os.path.join(root_dir, type)
    image_path = os.path.abspath(os.path.join(
        target_dir, 
        request.query.get("subfolder", ""), 
        request.query["filename"]
    ))
    c = os.path.commonpath((image_path, target_dir))
    if c != target_dir:
        return web.Response(status=403)

    if not os.path.isfile(image_path):
        return web.Response(status=404)

    return web.json_response(f"here is the prompt \n{image_path}")

class McBoaty_UpscalerRefiner_v5(McBoaty_Upscaler_v5, McBoaty_Refiner_v5):
    @classmethod
    def INPUT_TYPES(self):
        upscaler_inputs = McBoaty_Upscaler_v5.INPUT_TYPES()
        refiner_inputs = McBoaty_Refiner_v5.INPUT_TYPES()
        
        # Merge and deduplicate inputs
        combined_inputs = {**upscaler_inputs, **refiner_inputs}
        combined_inputs['required'] = {**upscaler_inputs['required'], **refiner_inputs['required']}
        combined_inputs['optional'] = {**upscaler_inputs.get('optional', {}), **refiner_inputs.get('optional', {})}
        combined_inputs['hidden'] = {"id":"UNIQUE_ID",}
        
        combined_inputs['required'].pop('pipe', None)
        combined_inputs['optional'].pop('pipe_prompty', None)
        
        combined_inputs['required'].pop('tiles_to_process', None)
        
        return combined_inputs

    RETURN_TYPES = (
        "MC_BOATY_PIPE",
        "MC_PROMPTY_PIPE_IN",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "STRING",
        "STRING"
    )
    
    RETURN_NAMES = (
        "McBoaty Pipe",
        "McPrompty Pipe",
        "image",
        "image (orignal)",
        "tiles",
        "prompts",
        "info"
    )
    
    OUTPUT_IS_LIST = (False,) * len(RETURN_TYPES)
    
    OUTPUT_NODE = True
    CATEGORY = "MaraScott/upscaling"
    DESCRIPTION = "An \"UPSCALER REFINER\" Node"
    FUNCTION = "fn"

    @classmethod
    def fn(self, **kwargs):
        start_time = time.time()
        
        # Upscaling phase
        upscaler_result = McBoaty_Upscaler_v5.fn(**kwargs)
        upscaler_pipe, _, upscaler_info = upscaler_result

        # Update kwargs with upscaler results for refiner
        kwargs.update({
            'pipe': upscaler_pipe,
        })

        # Refining phase
        refiner_result = McBoaty_Refiner_v5.fn(**kwargs)
        refiner_pipe, refiner_prompty_pipe, output_image, original_resized, output_tiles, grid_prompts, refiner_info = refiner_result

        end_time = time.time()
        total_time = int(end_time - start_time)

        # Combine info from both phases
        combined_info = self._combine_info(upscaler_info, refiner_info, total_time)

        return (
            refiner_pipe,
            refiner_prompty_pipe,
            output_image,
            original_resized,
            output_tiles,
            grid_prompts,
            combined_info
        )

    @staticmethod
    def _combine_info(upscaler_info, refiner_info, total_time):
        # Implement logic to combine info from upscaler and refiner
        combined_info = f"""
Upscaler Info:
{upscaler_info}

Refiner Info:
{refiner_info}

Total Execution Time: {total_time} seconds
"""
        return combined_info

    @classmethod
    def init(self, **kwargs):
        McBoaty_Upscaler_v5.init(**kwargs)
        McBoaty_Refiner_v5.init(**kwargs)