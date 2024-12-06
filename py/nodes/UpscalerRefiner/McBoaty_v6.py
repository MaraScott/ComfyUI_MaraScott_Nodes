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
from ...utils.constants import get_name, get_category

from ...utils.version import VERSION
from ...inc.lib.image import MS_Image_v2 as MS_Image
from ...vendor.ComfyUI_KJNodes.nodes.image_nodes import ColorMatch as ColorMatch
from ...inc.lib.llm import MS_Llm
from ...inc.lib.cache import MS_Cache

from ...utils.log import log, get_log, COLORS


class Mara_Common_v1():

    INPUTS = {}
    OUTPUTS = {}
    KSAMPLER = {}
    PARAMS = {}
    INFO = {}
    
    PIPE_ATTRIBUTES = ('INPUTS', 'PARAMS', 'KSAMPLER', 'OUTPUTS', 'INFO')

    @classmethod
    def set_pipe_values(self, pipe):
        for name, value in zip(self.PIPE_ATTRIBUTES, pipe):
            setattr(self, name, value)
    
class Mara_Tiler_v1(Mara_Common_v1):
    
    NAME = get_name('Image to tiles - v1')

    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "image": ("IMAGE", {"label": "Image" }),
                "upscale_model": (["None"]+folder_paths.get_filename_list("upscale_models"), { "label": "Upscale Model" }),
                "tile_size": ("INT", { "label": "Tile Size", "default": 512, "min": 320, "max": 4096, "step": 64}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = (
        "MC_BOATY_PIPE",
        "IMAGE",
    )
    
    RETURN_NAMES = (
        "McBoayty Pipe",
        "tiles",
    )
    
    OUTPUT_IS_LIST = (
        False,
        False,
    )
    
    
    OUTPUT_NODE = True
    DESCRIPTION = "An \"Tiler\" Node"
    FUNCTION = "fn"
    CATEGORY = get_category("Utils")

    @classmethod    
    def fn(self, **kwargs):
        
        start_time = time.time()

        self.init(**kwargs)
        
        log("McBoaty (Tiler) is starting to slicing the image", None, None, f"Node {self.INFO.id}")
        
        self.OUTPUTS.image, 
        self.INFO.is_image_divisible_by_8 = MS_Image().format_2_divby8(image=self.INPUTS.image)
        self.OUTPUTS.upscaled_image = self.OUTPUTS.image 
        if self.PARAMS.upscale_model is not None:
            self.OUTPUTS.upscaled_image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(self.PARAMS.upscale_model, self.OUTPUTS.upscaled_image)[0]
        self.INFO.image_width = self.OUTPUTS.upscaled_image.shape[1]
        self.INFO.image_height = self.OUTPUTS.upscaled_image.shape[2]

        self.OUTPUTS.tiles, self.PARAMS.grid_specs = self.get_tiles(image=self.OUTPUTS.upscaled_image)
        self.OUTPUTS.tiles = torch.cat(self.OUTPUTS.tiles)

        end_time = time.time()        
        self.INFO.execution_time = int(end_time - start_time)

        return (
            (
                self.INPUTS,
                self.OUTPUTS,
                self.PARAMS,
                self.INFO,
            ),
            self.OUTPUTS.tiles,
        )

    @classmethod
    def init(self, **kwargs):
        
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', None),
        )
        self.INPUTS = SimpleNamespace(
            image = kwargs.get('image', None),
        )
        self.OUTPUTS = SimpleNamespace(
            image = self.INPUTS.image,
            tiles = self.INPUTS.image,
        )

        if self.INPUTS.image is None:
            raise ValueError(f"{self.NAME} id {self.INFO.id}: No image provided")
        if not isinstance(self.INPUTS.image, torch.Tensor):
            raise ValueError(f"{self.NAME} id {self.INFO.id}: Image provided is not a Tensor")

        self.PARAMS = SimpleNamespace(
            upscale_model_name = kwargs.get('upscale_model', 'None'),
            upscale_model = None,
            upscale_model_scale = 1,
            tile_size = kwargs.get('tile_size', None),
            rows_qty = 1,
            cols_qty = 1,
        )
        if self.PARAMS.upscale_model_name != 'None':
            self.PARAMS.upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader().load_model(self.PARAMS.upscale_model_name)[0]
            self.PARAMS.upscale_model_scale = self.PARAMS.upscale_model.scale
    
    @classmethod
    def get_tiles(self, image):
        
        rows_qty_float = (image.shape[1]) / self.PARAMS.tile_size
        cols_qty_float = (image.shape[2]) / self.PARAMS.tile_size
        rows_qty = math.ceil(rows_qty_float)
        cols_qty = math.ceil(cols_qty_float)

        tiles_qty = rows_qty * cols_qty        
        if tiles_qty > 16384 :
            msg = get_log(f"\n\n--------------------\n\n!!! Number of tiles is higher than 16384 ({tiles_qty} for {self.PARAMS.cols_qty} cols and {self.PARAMS.rows_qty} rows)!!!\n\nPlease consider increasing your tile and feather sizes\n\n--------------------\n", "BLUE", "YELLOW", f"Node {self.INFO.id} - {self.NAME}")
            raise ValueError(msg)

        self.PARAMS.rows_qty = rows_qty
        self.PARAMS.cols_qty = cols_qty
        
        grid_specs = MS_Image().get_tiled_grid_specs(image, self.PARAMS.tile_size, self.PARAMS.rows_qty, self.PARAMS.cols_qty, 0)[0]
        grid_images = MS_Image().get_grid_images(image, grid_specs)
        
        return grid_images, grid_specs
        
        
class Mara_Untiler_v1(Mara_Common_v1):
    
    NAME = get_name('Tiles to Image - v1')
    
    UPSCALE_METHODS = [
        "area", 
        "bicubic", 
        "bilinear", 
        "bislerp",
        "lanczos",
        "nearest-exact"
    ]
        
    INPUTS = {}
    OUTPUTS = {}
    PARAMS = {}
    INFO = {}
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "pipe": ("MC_BOATY_PIPE", {"label": "McBoaty Pipe" }),
                "tiles": ("IMAGE", {"label": "Image" }),
                "output_upscale_method": (self.UPSCALE_METHODS, { "label": "Custom Output Upscale Method", "default": "bicubic"}),
                "output_size_type": ("BOOLEAN", { "label": "Output Size Type", "default": True, "label_on": "Upscale size", "label_off": "Custom size"}),
                "output_size": ("FLOAT", { "label": "Custom Output Size", "default": 1.00, "min": 1.00, "max": 16.00, "step":0.01, "round": 0.01}),
                
            },
            "optional": {
            }
        }

    RETURN_TYPES = (
        "IMAGE",
    )
    
    RETURN_NAMES = (
        "image",
    )
    
    OUTPUT_IS_LIST = (
        False,
    )
    
    
    OUTPUT_NODE = True
    DESCRIPTION = "An \"Untiler\" Node"
    FUNCTION = "fn"
    CATEGORY = get_category("Utils")

    @classmethod    
    def fn(self, **kwargs):
        
        start_time = time.time()

        self.init(**kwargs)
        
        log("McBoaty (Untiler) is starting to rebuild the image", None, None, f"Node {self.INFO.id}")
        
        # feather_mask = 64 # self.PARAMS.feather_mask
        # upscale_model_scale = 1 # self.PARAMS.upscale_model_scale
        # grid_prompts = [] # self.OUTPUTS.grid_prompts
        
        # self.OUTPUTS.image, tiles_order = MS_Image().rebuild_image_from_parts(
        #     0, 
        #     self.INPUTS.tiles, 
        #     self.OUTPUTS.image, 
        #     self.PARAMS.grid_specs, 
        #     feather_mask, 
        #     upscale_model_scale, 
        #     self.PARAMS.rows_qty, 
        #     self.PARAMS.cols_qty, 
        #     grid_prompts
        # )

        # if not self.PARAMS.upscale_size_type:
        #     self.OUTPUTS.image = nodes.ImageScale().upscale(self.OUTPUTS.image, self.PARAMS.upscale_method, int(self.OUTPUTS.image.shape[2] * self.PARAMS.upscale_size), int(self.OUTPUTS.image.shape[1] * self.PARAMS.upscale_size), False)[0]

        end_time = time.time()        
        self.INFO.execution_time = int(end_time - start_time)

        return (
            self.OUTPUTS.image,
        )
        
    @classmethod
    def init(self, **kwargs):
        
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', None),
        )

        pipe = kwargs.get('pipe', (None,) * len(self.PIPE_ATTRIBUTES))
        self.set_pipe_values(pipe)
        
        self.INPUTS = SimpleNamespace(
            tiles = kwargs.get('tiles', None),
        )
        self.OUTPUTS = SimpleNamespace(
            image = self.INPUTS.tiles,
            tiles = self.INPUTS.tiles,
        )

        self.PARAMS.upscale_size_type = kwargs.get('output_size_type', None)
        self.PARAMS.upscale_size = kwargs.get('output_size', None)
        self.PARAMS.upscale_method = kwargs.get('output_upscale_method', "lanczos"),

        if self.INPUTS.tiles is None:
            raise ValueError(f"{self.NAME} id {self.INFO.id}: No image provided")
        if not isinstance(self.INPUTS.tiles, torch.Tensor):
            raise ValueError(f"{self.NAME} id {self.INFO.id}: Image provided is not a Tensor")
        
class Mara_McBoaty_Configurator_v6(Mara_Common_v1):

    
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
    
    INPUTS = {}
    OUTPUTS = {}
    PARAMS = {}
    INFOS = {}
    KSAMPLERS = {}
    LLM = {}
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "pipe": ("MC_BOATY_PIPE", {"label": "McBoaty Pipe" }),
                "tiles": ("IMAGE", {"label": "Tiles" }),

                "model": ("MODEL", { "label": "Model" }),
                "clip": ("CLIP", { "label": "Clip" }),
                "vae": ("VAE", { "label": "VAE" }),
                "positive": ("STRING", { "label": "Positive (Prompt)", "multiline": True }),
                "negative": ("STRING", { "label": "Negative (Prompt)", "multiline": True }),
                "sigmas_type": (self.SIGMAS_TYPES, { "label": "Sigmas Type" }),
                "model_type": (self.MODEL_TYPES, { "label": "Model Type", "default": "SDXL" }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, { "label": "Sampler Name" }),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, { "label": "Basic Scheduler" }),
                "steps": ("INT", { "label": "Steps", "default": 10, "min": 1, "max": 10000}),
                "seed": ("INT", { "label": "Seed", "default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", { "label": "CFG", "default": 2.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "denoise": ("FLOAT", { "label": "Denoise", "default": 0.27, "min": 0.0, "max": 1.0, "step": 0.01}),

                "vae_encode": ("BOOLEAN", { "label": "VAE Encode type", "default": True, "label_on": "tiled", "label_off": "standard"}),
                "tile_size_vae": ("INT", { "label": "Tile Size (VAE)", "default": 512, "min": 256, "max": 4096, "step": 64}),


            },
            "optional": {
                "Florence2": ("FL2MODEL", { "label": "Florence2" }),
                "Llm_party": ("CUSTOM", { "label": "LLM Model" }),
            }
        }

    RETURN_TYPES = (
        "MC_BOATY_PIPE", 
        "MC_PROMPTY_PIPE",
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
    CATEGORY = get_category('Upscaling/v6')
    DESCRIPTION = "An \"UPSCALER\" Node"
    FUNCTION = "fn"

    @classmethod    
    def fn(self, **kwargs):

        start_time = time.time()
        
        self.init(**kwargs)
        
        log("McBoaty (Upscaler) is starting to do its magic", None, None, f"Node {self.INFO.id}")
                
        # self.PARAMS.grid_specs, self.OUTPUTS.grid_images, self.OUTPUTS.grid_prompts = self.upscale(self.INPUTS.tiles, "Upscaling")

        # self.OUTPUTS.tiles = torch.cat(self.OUTPUTS.grid_images)
        self.OUTPUTS.tiles = self.INPUTS.tiles
        
        end_time = time.time()
        output_info = self._get_info(
            0, # self.INFO.image_width, 
            0, # self.INFO.image_height, 
            True, # self.INFO.image_divisible_by_8, 
            int(end_time - start_time)
        )
        
        log("McBoaty (Upscaler) is done with its magic", None, None, f"Node {self.INFO.id}")


        return (
            (
                self.INPUTS,
                self.PARAMS,
                self.KSAMPLER,
                self.OUTPUTS,
            ),
            (
                self.OUTPUTS.tiles
            ),
            output_info
        )
        
    @classmethod
    def init(self, **kwargs):
        # Initialize the bus tuple with None values for each parameter
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', None),
        )
        
        pipe = kwargs.get('pipe', (None,) * len(self.PIPE_ATTRIBUTES))
        self.set_pipe_values(pipe)
        
        self.INPUTS = SimpleNamespace(
            tiles = kwargs.get('tiles', None),
        )
        if self.INPUTS.tiles is None:
            raise ValueError(f"{self.NAME} id {self.INFO.id}: No tiles provided")

        if not isinstance(self.INPUTS.tiles, torch.Tensor):
            raise ValueError(f"{self.NAME} id {self.INFO.id}: tiles provided are not Tensors")
                
        self.LLM = SimpleNamespace(
            vision_model = kwargs.get('FL2MODEL', None),
            model = kwargs.get('llm_model', None),
        )
        
        self.PARAMS = SimpleNamespace(
            tile_prompting_active = kwargs.get('tile_prompting_active', False),
            grid_specs = None,
            rows_qty = 1,
            cols_qty = 1,
        )

        self.KSAMPLER = SimpleNamespace(
            tiled = kwargs.get('vae_encode', None),
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
            cfg = kwargs.get('cfg', None),
            denoise = kwargs.get('denoise', None),
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
    def _get_info(self, image_width, image_height, image_divisible_by_8, execution_duration):
        
        return [f"""

    IMAGE (INPUT)
        width   :   {image_width}
        height  :   {image_height}
        image divisible by 8 : {image_divisible_by_8}

    ------------------------------

    ------------------------------
    
    EXECUTION
        DURATION : {execution_duration} seconds

    NODE INFO
        version : {VERSION}

"""]        
    
    @classmethod
    def upscale(self, image, iteration):
        
        feather_mask = self.PARAMS.feather_mask
        rows_qty_float = (image.shape[1] * self.PARAMS.upscale_model_scale) / self.KSAMPLER.tile_size
        cols_qty_float = (image.shape[2] * self.PARAMS.upscale_model_scale) / self.KSAMPLER.tile_size
        rows_qty = math.ceil(rows_qty_float)
        cols_qty = math.ceil(cols_qty_float)

        tiles_qty = rows_qty * cols_qty        
        if tiles_qty > 16384 :
            msg = get_log(f"\n\n--------------------\n\n!!! Number of tiles is higher than 16384 ({tiles_qty} for {self.PARAMS.cols_qty} cols and {self.PARAMS.rows_qty} rows)!!!\n\nPlease consider increasing your tile and feather sizes\n\n--------------------\n", "BLUE", "YELLOW", f"Node {self.INFO.id} - Mara_McBoaty_Configurator_v6")
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

class Mara_McBoaty_Refiner_v6(Mara_Common_v1):
    
    NAME = get_name('McBoaty Refiner - v6')

    COLOR_MATCH_METHODS = [   
        'none',
        'mkl',
        'hm', 
        'reinhard', 
        'mvgd', 
        'hm-mvgd-hm', 
        'hm-mkl-hm',
    ]
    
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
            },
            "optional": {
                "pipe_prompty": ("MC_PROMPTY_PIPE", {"label": "McPrompty Pipe" }),
                "color_match_method": (self.COLOR_MATCH_METHODS, { "label": "Color Match Method", "default": 'none'}),
            }
        }

    RETURN_TYPES = (
        "MC_BOATY_PIPE", 
        "MC_PROMPTY_PIPE", 
        "IMAGE",
        "STRING"
    )
    
    RETURN_NAMES = (
        "McBoaty Pipe", 
        "McPrompty Pipe",
        "tiles", 
        "info", 
    )
    
    OUTPUT_IS_LIST = (False,) * len(RETURN_TYPES)
    
    
    OUTPUT_NODE = True
    CATEGORY = get_category('Upscaling/v6')
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
        
        # PARAMS.grid_prompts, OUTPUTS.output_tiles, OUTPUTS.grid_tiles_to_process = self.refine(self.OUTPUTS.image, "Upscaling")
                
        # output_tiles = torch.cat(self.OUTPUTS.grid_images)

        # if self.PARAMS.color_match_method != 'none':
        #     self.OUTPUTS.image = ColorMatch().colormatch(self.INPUTS.image, self.OUTPUTS.image, self.PARAMS.color_match_method)[0]

        end_time = time.time()

        output_info = self._get_info(
            int(end_time - start_time)
        )

        log("McBoaty (Refiner) is done with its magic", None, None, f"Node {self.INFO.id}")
        
        return (
            (
                INPUTS,
                PARAMS,
                KSAMPLER,
                OUTPUTS,
            ),
            (
                self.OUTPUTS.tiles,
            ),            
            self.OUTPUTS.tiles, 
            output_info, 
        )
        
    @classmethod
    def init(self, **kwargs):
        
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', None),
        )
        
        pipe = kwargs.get('pipe', (None,) * len(self.PIPE_ATTRIBUTES))
        self.set_pipe_values(pipe)

        _tiles_to_process = kwargs.get('tiles_to_process', '')
        self.PARAMS.tiles_to_process = self.set_tiles_to_process(_tiles_to_process)
        self.PARAMS.color_match_method = kwargs.get('color_match_method', 'none'),
        
        self.KSAMPLER.sampler_name = kwargs.get('sampler_name', None)
        self.KSAMPLER.scheduler = kwargs.get('basic_scheduler', None)
        self.KSAMPLER.sigmas_type = kwargs.get('sigmas_type', None)
        self.KSAMPLER.model_type = kwargs.get('model_type', None)
        self.KSAMPLER.steps = kwargs.get('steps', None)
                
        # self.KSAMPLER.sampler = comfy_extras.nodes_custom_sampler.KSamplerSelect().get_sampler(self.KSAMPLER.sampler_name)[0]
        # self.KSAMPLER.tile_size_sampler = self.MODEL_TYPE_SIZES[self.KSAMPLER.model_type]
        # self.KSAMPLER.sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, self.KSAMPLER.denoise, self.KSAMPLER.scheduler, self.KSAMPLER.model_type)
        # self.KSAMPLER.outpaint_sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, 1, self.KSAMPLER.scheduler, self.KSAMPLER.model_type)

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
        
    @classmethod    
    def _get_sigmas(self, sigmas_type, model, steps, denoise, scheduler, model_type):
        if sigmas_type == "SDTurboScheduler":
            SigmaScheduler = getattr(comfy_extras.nodes_custom_sampler, sigmas_type)
            sigmas = SigmaScheduler().get_sigmas(model, steps, denoise)[0]
        elif sigmas_type == "AlignYourStepsScheduler":
            SigmaScheduler = AlignYourStepsScheduler
            if model_type == "SD3" or model_type == "FLUX1":
                model_type = "SDXL"
            sigmas = SigmaScheduler().get_sigmas(model_type, steps, denoise)[0]
        else: # BasicScheduler
            SigmaScheduler = getattr(comfy_extras.nodes_custom_sampler, sigmas_type)
            sigmas = SigmaScheduler().get_sigmas(model, scheduler, steps, denoise)[0]
        
        return sigmas    
    
            
    @classmethod
    def set_tiles_to_process(self, tiles_to_process=''):

        max_tiles = len(self.OUTPUTS.grid_tiles_to_process)
        max = max_tiles if max_tiles > 0 else 16384
        
        def is_valid_index(index, max = 16384):
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
                    output = (nodes.VAEDecodeTiled().decode(self.KSAMPLER.vae, latent_output, self.KSAMPLER.tile_size_vae, 0)[0].unsqueeze(0))[0]
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

        # _tiles_order = tuple(output for _, output, _ in tiles_order)
        # tiles_order.sort(key=lambda x: x[0])
        # output_tiles = tuple(output for _, output, _ in tiles_order)
        # output_tiles = torch.cat(output_tiles)
        output_tiles = output
        # output_prompts = tuple(prompt for _, _, prompt in tiles_order)
        output_prompts = tuple("")

        return output_prompts, output_tiles

class Mara_McBoaty_TilePrompter_v6(Mara_Common_v1):

    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "pipe": ("MC_PROMPTY_PIPE", {"label": "McPrompty Pipe" }),
                "tiles_to_process": ("STRING", { "label": "Tile to process", "default": ''}),
                "positive": ("STRING", { "label": "Positive (Prompt)", "multiline": True }),
                "negative": ("STRING", { "label": "Negative (Prompt)", "multiline": True }),
                "cfg": ("FLOAT", { "label": "CFG", "default": 2.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "denoise": ("FLOAT", { "label": "Denoise", "default": 0.27, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (
        "MC_PROMPTY_PIPE",
    )
    
    RETURN_NAMES = (
        "McPrompty Pipe",
    )
    
    OUTPUT_IS_LIST = (
        False,
    )
        
    OUTPUT_NODE = True
    CATEGORY = get_category('Upscaling/v6')
    DESCRIPTION = "A \"Tile Prompt Editor\" Node"
    FUNCTION = "fn"

    @classmethod    
    def fn(self, **kwargs):
                        
        input_tiles = kwargs.get('pipe', (None, None))

        self.init(**kwargs)
        
        log("McBoaty (PromptEditor) is starting to do its magic", None, None, f"Node {self.INFO.id}")
        

        log("McBoaty (PromptEditor) is done with its magic", None, None, f"Node {self.INFO.id}")
                    
        return (
            (
                self.INPUTS,
                self.PARAMS,
                self.KSAMPLER,
                self.OUTPUTS,
            )
        )

    @classmethod
    def init(self, **kwargs):
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', 0),
        )
        
        self.output_dir = folder_paths.get_temp_directory()

class Mara_McBoaty_v6(Mara_McBoaty_Configurator_v6, Mara_McBoaty_Refiner_v6):

    @classmethod
    def INPUT_TYPES(self):
        upscaler_inputs = Mara_McBoaty_Configurator_v6.INPUT_TYPES()
        refiner_inputs = Mara_McBoaty_Refiner_v6.INPUT_TYPES()
        
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
        "MC_PROMPTY_PIPE",
        "IMAGE",
        "STRING"
    )
    
    RETURN_NAMES = (
        "McBoaty Pipe",
        "McPrompty Pipe",
        "tiles",
        "info"
    )
    
    OUTPUT_IS_LIST = (False,) * len(RETURN_TYPES)
    
    OUTPUT_NODE = True
    CATEGORY = get_category('Upscaling/v6')
    DESCRIPTION = "An \"UPSCALER REFINER\" Node"
    FUNCTION = "fn"

    @classmethod
    def fn(self, **kwargs):
        
        start_time = time.time()
        
        self.INFO = SimpleNamespace(
            id = kwargs.get('id', None),
        )        
        
        # Upscaling phase
        upscaler_result = Mara_McBoaty_Configurator_v6().fn(**kwargs)
        upscaler_pipe, _, upscaler_info = upscaler_result

        # Update kwargs with upscaler results for refiner
        kwargs.update({
            'pipe': upscaler_pipe,
        })

        # Refining phase
        refiner_pipe, refiner_prompty_pipe, output_image, original_resized, output_tiles, grid_prompts, refiner_info = Mara_McBoaty_Refiner_v6.fn(**kwargs)

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
        Mara_McBoaty_Configurator_v6.init(**kwargs)
        Mara_McBoaty_Refiner_v6.init(**kwargs)