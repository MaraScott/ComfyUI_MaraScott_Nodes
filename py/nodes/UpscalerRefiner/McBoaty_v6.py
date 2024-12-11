#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Upscaler Refiner Node.  Upscale and Refine a picture by 2 using a 9 Square Grid to upscale and refine the visual in 9 sequences
#
###

import os
import sys
import time
import copy
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

    NAME = get_name('🐰 McBoaty Set - v6')
    
    MAX_TILES = 16384

    PIPE_ATTRIBUTES = (
        'INPUTS', 
        'OUTPUTS', 
        'PARAMS', 
        'INFO', 

        'CONTROLNET',
        'KSAMPLER', 
        'LLM', 
    )
    
    TILE_ATTRIBUTES = SimpleNamespace(
        positive = '',
        negative = '',
        cfg = 2.5,
        denoise = 0.27,
    )
    
    def __init__(self):
        # Dynamically create attributes from PIPE_ATTRIBUTES
        for attr in self.PIPE_ATTRIBUTES:
            if not hasattr(type(self), attr):  # Only set the attribute if it doesn't already exist
                setattr(type(self), attr, SimpleNamespace())

    @classmethod
    def set_pipe_values(self, pipe):
        for name, value in zip(self.PIPE_ATTRIBUTES, pipe):
            setattr(self, name, value)

    @classmethod
    def set_mc_boaty_pipe(self):
        return tuple(getattr(self, attr, None) for attr in self.PIPE_ATTRIBUTES)
    
    @classmethod
    def parse_tiles_to_process(self, tiles_to_process = "", MAX_TILES = 16384):
        result = set()  # Initialize an empty set for results
        
        if not tiles_to_process or tiles_to_process.strip() == '':
            return []  # Handle empty or invalid input immediately

        try:
            # Split by comma to handle numbers and ranges
            parts = tiles_to_process.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:  # Handle ranges
                    try:
                        range_parts = part.split('-')
                        if len(range_parts) != 2:
                            continue  # Skip invalid ranges
                        start, end = map(int, range_parts)
                        if start > end:
                            start, end = end, start  # Swap if range is reversed
                        result.update(num for num in range(start, end + 1) if 1 <= num <= MAX_TILES)
                    except ValueError:
                        continue  # Skip invalid ranges
                else:  # Handle single numbers
                    try:
                        num = int(part)
                        if 1 <= num <= MAX_TILES:  # Ignore out-of-range numbers
                            result.add(num)
                    except ValueError:
                        continue  # Skip invalid numbers

        except Exception:
            pass  # Ignore unexpected errors but allow processing to continue

        # Return a sorted list of unique valid values
        return sorted(result)

    @classmethod
    def override_tiles(self, tiles, new_tiles):
        
        for index, tile in enumerate(tiles):
            if index >= len(new_tiles):
                continue  # Skip if the index doesn't exist in the `tiles` list
            
            if len(self.PARAMS.tiles_to_process) == 0 or index in self.PARAMS.tiles_to_process:
                override_tile = new_tiles[index]

                # Compare attributes and override only if they differ
                for attr in vars(override_tile):  # Loop through attributes of the override tile
                    override_value = getattr(override_tile, attr)
                    if hasattr(tile, attr):  # Only override if the attribute exists in the master
                        tile_value = getattr(tile, attr)
                        override_value = getattr(override_tile, attr)
                        if isinstance(tile_value, torch.Tensor) and isinstance(override_value, torch.Tensor):
                            if not torch.equal(tile_value, override_value):
                                setattr(tiles[index], attr, override_value)                
                        else:
                            if tile_value != override_value:
                                setattr(tiles[index], attr, override_value)                
        return tiles

    
class Mara_Tiler_v1(Mara_Common_v1):
    
    NAME = get_name('🐰 Image to tiles - v1')

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
        
        self.OUTPUTS.image, _, _, self.INFO.is_image_divisible_by_8 = MS_Image().format_2_divby8(image=self.INPUTS.image)
        self.OUTPUTS.upscaled_image = self.OUTPUTS.image 
        if self.PARAMS.upscale_model is not None:
            self.OUTPUTS.upscaled_image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(self.PARAMS.upscale_model, self.OUTPUTS.upscaled_image)[0]
        self.INFO.image_width = self.OUTPUTS.upscaled_image.shape[1]
        self.INFO.image_height = self.OUTPUTS.upscaled_image.shape[2]

        self.OUTPUTS.tiles, self.PARAMS.grid_specs = self.get_tiles(image=self.OUTPUTS.upscaled_image)
        
        tiles = []
        for index, tile in enumerate(self.OUTPUTS.tiles):
            _tile = copy.deepcopy(self.TILE_ATTRIBUTES)
            _tile.id = index + 1
            _tile.tile = tile
            tiles.append(_tile)
            
        self.KSAMPLER.tiles = tiles

        end_time = time.time()        
        self.INFO.execution_time = int(end_time - start_time)
        
        mc_boaty_pipe = self.set_mc_boaty_pipe()
        
        self.OUTPUTS.tiles = torch.cat([t.tile for t in self.KSAMPLER.tiles], dim=0)
        
        return (
            mc_boaty_pipe,
            self.OUTPUTS.tiles,
        )

    @classmethod
    def init(self, **kwargs):
        
        self.INFO.id = kwargs.get('id', None)
        self.INPUTS.image = kwargs.get('image', None)
        self.OUTPUTS.image = self.INPUTS.image
        self.OUTPUTS.tiles = self.INPUTS.image

        if self.INPUTS.image is None:
            raise ValueError(f"{self.NAME} id {self.INFO.id}: No image provided")
        if not isinstance(self.INPUTS.image, torch.Tensor):
            raise ValueError(f"{self.NAME} id {self.INFO.id}: Image provided is not a Tensor")

        self.PARAMS.upscale_model_name = kwargs.get('upscale_model', 'None')
        self.PARAMS.upscale_model = None
        self.PARAMS.upscale_model_scale = 1
        self.PARAMS.tile_size = kwargs.get('tile_size', None)
        self.PARAMS.rows_qty = 1
        self.PARAMS.cols_qty = 1

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
        if tiles_qty > self.MAX_TILES :
            msg = get_log(f"\n\n--------------------\n\n!!! Number of tiles is higher than {self.MAX_TILES} ({tiles_qty} for {self.PARAMS.cols_qty} cols and {self.PARAMS.rows_qty} rows)!!!\n\nPlease consider increasing your tile and feather sizes\n\n--------------------\n", "BLUE", "YELLOW", f"Node {self.INFO.id} - {self.NAME}")
            raise ValueError(msg)

        self.PARAMS.rows_qty = rows_qty
        self.PARAMS.cols_qty = cols_qty
        
        grid_specs = MS_Image().get_dynamic_grid_specs(image.shape[2], image.shape[1], self.PARAMS.rows_qty, self.PARAMS.cols_qty, 0)[0]
        # grid_specs = MS_Image().get_tiled_grid_specs(image, self.PARAMS.tile_size, self.PARAMS.rows_qty, self.PARAMS.cols_qty, 0)[0]
        grid_images = MS_Image().get_grid_images(image, grid_specs)
        
        return grid_images, grid_specs
        
        
class Mara_Untiler_v1(Mara_Common_v1):
    
    NAME = get_name('🐰 Tiles to Image - v1')
    
    UPSCALE_METHODS = [
        "area", 
        "bicubic", 
        "bilinear", 
        "bislerp",
        "lanczos",
        "nearest-exact"
    ]

    UPSCALE_SIZE_REF = [
        "Output Image",
        "Input Image",
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
                "output_upscale_method": (self.UPSCALE_METHODS, { "label": "Custom Output Upscale Method", "default": "bicubic"}),
                "output_size_ref": (self.UPSCALE_SIZE_REF, { "label": "Output Size Ref", "default": "Output Image"}),
                "output_size": ("FLOAT", { "label": "Custom Output Size", "default": 1.00, "min": 0.10, "max": 16.00, "step":0.01, "round": 0.01}),
                
            },
            "optional": {
                "tiles": ("IMAGE", {"label": "Image" }),
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
        
        self.OUTPUTS.image, tiles_order = MS_Image().rebuild_image_from_parts(
            0, 
            self.OUTPUTS.tiles, 
            self.OUTPUTS.image, 
            self.PARAMS.grid_specs, 
            self.PARAMS.feather_mask, 
            self.PARAMS.upscale_model_scale, 
            self.PARAMS.rows_qty, 
            self.PARAMS.cols_qty, 
            self.OUTPUTS.grid_prompts
        )

        if not (self.PARAMS.upscale_size_ref == self.UPSCALE_SIZE_REF[0] and self.PARAMS.upscale_size == 1.00):
            image_ref = self.OUTPUTS.image
            if self.PARAMS.upscale_size_ref != self.UPSCALE_SIZE_REF[0]:
                image_ref = self.INPUTS.image
            self.OUTPUTS.image = nodes.ImageScale().upscale(self.OUTPUTS.image, self.PARAMS.upscale_method, int(image_ref.shape[2] * self.PARAMS.upscale_size), int(image_ref.shape[1] * self.PARAMS.upscale_size), False)[0]

        # log((self.INPUTS.image, self.OUTPUTS.image))
        # if self.PARAMS.color_match_method != 'none':
        #     self.OUTPUTS.image = ColorMatch().colormatch(self.INPUTS.image, self.OUTPUTS.image, self.PARAMS.color_match_method)[0]

        end_time = time.time()
        self.INFO.execution_time = int(end_time - start_time)

        return (
            self.OUTPUTS.image,
        )
        
    @classmethod
    def init(self, **kwargs):

        pipe = kwargs.get('pipe', (SimpleNamespace(),) * len(self.PIPE_ATTRIBUTES))
        self.set_pipe_values(pipe)

        self.INFO.id = kwargs.get('id', None)
        self.INPUTS.tiles = kwargs.get('tiles', None)
        self.OUTPUTS.image = self.INPUTS.image

        self.PARAMS.upscale_size_ref = kwargs.get('output_size_ref', False)
        self.PARAMS.upscale_size = kwargs.get('output_size', 1.00)
        self.PARAMS.upscale_method = kwargs.get('output_upscale_method', "lanczos")
        
class Mara_McBoaty_Configurator_v6(Mara_Common_v1):

    NAME = get_name('🐰 McBoaty Configurator - v6')
    
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
        
    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "pipe": ("MC_BOATY_PIPE", {"label": "McBoaty Pipe" }),

                "model": ("MODEL", { "label": "Model" }),
                "clip": ("CLIP", { "label": "Clip" }),
                "vae": ("VAE", { "label": "VAE" }),
                "positive": ("STRING", { "label": "Positive (Prompt)", "multiline": True, "default": "" }),
                "negative": ("STRING", { "label": "Negative (Prompt)", "multiline": True, "default": "" }),
                "sigmas_type": (self.SIGMAS_TYPES, { "label": "Sigmas Type" }),
                "model_type": (self.MODEL_TYPES, { "label": "Model Type", "default": "SDXL" }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, { "label": "Sampler Name" }),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, { "label": "Basic Scheduler" }),
                "steps": ("INT", { "label": "Steps", "default": 10, "min": 1, "max": 10000}),
                "seed": ("INT", { "label": "Seed", "default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", { "label": "CFG", "default": self.TILE_ATTRIBUTES.cfg, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "denoise": ("FLOAT", { "label": "Denoise", "default": self.TILE_ATTRIBUTES.denoise, "min": 0.0, "max": 1.0, "step": 0.01}),

                "vae_encode": ("BOOLEAN", { "label": "VAE Encode type", "default": True, "label_on": "tiled", "label_off": "standard"}),
                "tile_size_vae": ("INT", { "label": "Tile Size (VAE)", "default": 512, "min": 256, "max": 4096, "step": 64}),


            },
            "optional": {
                "tiles": ("IMAGE", {"label": "Tiles" }),
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
        
        for tile in self.KSAMPLER.tiles:
            tile.positive = self.KSAMPLER.positive
            tile.negative = self.KSAMPLER.negative
            tile.cfg = self.KSAMPLER.cfg
            tile.denoise = self.KSAMPLER.denoise

        end_time = time.time()

        output_info = self._get_info(
            0, # self.INFO.image_width, 
            0, # self.INFO.image_height, 
            True, # self.INFO.image_divisible_by_8, 
            int(end_time - start_time)
        )
        
        mc_boaty_pipe = self.set_mc_boaty_pipe()
        
        log("McBoaty (Upscaler) is done with its magic", None, None, f"Node {self.INFO.id}")

        return (
            mc_boaty_pipe,
            (
                self.KSAMPLER.tiles,
            ),
            output_info
        )
        
    @classmethod
    def init(self, **kwargs):

        pipe = kwargs.get('pipe', (SimpleNamespace(),) * len(self.PIPE_ATTRIBUTES))
        self.set_pipe_values(pipe)

        self.INFO.id = kwargs.get('id', None)
        
        self.INPUTS.tiles = kwargs.get('tiles', None)
        if self.INPUTS.tiles is not None and not isinstance(self.INPUTS.tiles, torch.Tensor):
            raise ValueError(f"{self.NAME} id {self.INFO.id}: tiles provided are not Tensors")
                
        self.LLM.vision_model = kwargs.get('FL2MODEL', None)
        self.LLM.model = kwargs.get('llm_model', None)
        
        self.PARAMS.tile_prompting_active = kwargs.get('tile_prompting_active', False)

        self.KSAMPLER.positive = kwargs.get('positive', '')
        self.KSAMPLER.negative = kwargs.get('negative', '')
        self.KSAMPLER.tiled = kwargs.get('vae_encode', None)
        self.KSAMPLER.tile_size_vae = kwargs.get('tile_size_vae', None)
        self.KSAMPLER.model = kwargs.get('model', None)
        self.KSAMPLER.clip = kwargs.get('clip', None)
        self.KSAMPLER.vae = kwargs.get('vae', None)
        self.KSAMPLER.noise_seed = kwargs.get('seed', None)
        self.KSAMPLER.add_noise = True

        self.KSAMPLER.sigmas_type = kwargs.get('sigmas_type', None)
        self.KSAMPLER.model_type = kwargs.get('model_type', None)
        self.KSAMPLER.tile_size_sampler = self.MODEL_TYPE_SIZES[self.KSAMPLER.model_type]
        self.KSAMPLER.sampler_name = kwargs.get('sampler_name', None)
        self.KSAMPLER.sampler = comfy_extras.nodes_custom_sampler.KSamplerSelect().get_sampler(self.KSAMPLER.sampler_name)[0]
        self.KSAMPLER.scheduler = kwargs.get('basic_scheduler', None)
        self.KSAMPLER.steps = kwargs.get('steps', None)
        self.KSAMPLER.positive = kwargs.get('positive', '')
        self.KSAMPLER.negative = kwargs.get('negative', '')
        self.KSAMPLER.cfg = kwargs.get('cfg', self.TILE_ATTRIBUTES.cfg)
        self.KSAMPLER.denoise = kwargs.get('denoise', self.TILE_ATTRIBUTES.denoise)
        
        self.KSAMPLER.sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, self.KSAMPLER.denoise, self.KSAMPLER.scheduler, self.KSAMPLER.model_type)
        # self.KSAMPLER.outpaint_sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, 1, self.KSAMPLER.scheduler, self.KSAMPLER.model_type)

        self.KSAMPLER.control_net_name = None
        self.KSAMPLER.control = None

        # TODO : make the feather_mask proportional to tile size ?
        # self.PARAMS.feather_mask = self.KSAMPLER.tile_size // 16
        self.PARAMS.feather_mask = 0

        self.OUTPUTS.grid_images = []
        self.OUTPUTS.grid_prompts = [self.KSAMPLER.positive for _ in self.PARAMS.grid_specs]
        self.OUTPUTS.output_info = ["No info"]
        self.OUTPUTS.grid_tiles_to_process = []
    
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
        
        rows_qty_float = (image.shape[1] * self.PARAMS.upscale_model_scale) / self.KSAMPLER.tile_size
        cols_qty_float = (image.shape[2] * self.PARAMS.upscale_model_scale) / self.KSAMPLER.tile_size
        rows_qty = math.ceil(rows_qty_float)
        cols_qty = math.ceil(cols_qty_float)

        tiles_qty = rows_qty * cols_qty        
        if tiles_qty > self.MAX_TILES :
            msg = get_log(f"\n\n--------------------\n\n!!! Number of tiles is higher than {self.MAX_TILES} ({tiles_qty} for {self.PARAMS.cols_qty} cols and {self.PARAMS.rows_qty} rows)!!!\n\nPlease consider increasing your tile and feather sizes\n\n--------------------\n", "BLUE", "YELLOW", f"Node {self.INFO.id} - Mara_McBoaty_Configurator_v6")
            raise ValueError(msg)

        upscaled_image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(self.PARAMS.upscale_model, image)[0]

        self.PARAMS.rows_qty = rows_qty
        self.PARAMS.cols_qty = cols_qty
        
        
        # grid_specs = MS_Image().get_dynamic_grid_specs(upscaled_image.shape[2], upscaled_image.shape[1], rows_qty, cols_qty, self.PARAMS.feather_mask)[0]
        grid_specs = MS_Image().get_tiled_grid_specs(upscaled_image, self.KSAMPLER.tile_size, self.PARAMS.rows_qty, self.PARAMS.cols_qty, self.PARAMS.feather_mask)[0]
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
    
    NAME = get_name('🐰 McBoaty Refiner - v6')

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

        tiles = kwargs.get('pipe_prompty', ([],))[0]
        tiles = self.refine(tiles, "Upscaling")
        self.KSAMPLER.tiles = self.override_tiles(self.KSAMPLER.tiles, tiles)
        end_time = time.time()
        

        output_info = self._get_info(
            int(end_time - start_time)
        )

        mc_boaty_pipe = self.set_mc_boaty_pipe()
        
        self.OUTPUTS.tiles = torch.cat([t.tile for t in self.KSAMPLER.tiles], dim=0)

        log("McBoaty (Refiner) is done with its magic", None, None, f"Node {self.INFO.id}")

        return (
            mc_boaty_pipe,
            (
                self.KSAMPLER.tiles,
            ),            
            self.OUTPUTS.tiles,
            output_info, 
        )
        
    @classmethod
    def init(self, **kwargs):
        
        pipe = kwargs.get('pipe', (SimpleNamespace(),) * len(self.PIPE_ATTRIBUTES))
        self.set_pipe_values(pipe)

        self.INFO.id = kwargs.get('id', None)        

        _tiles_to_process = kwargs.get('tiles_to_process', '')
        self.PARAMS.tiles_to_process = self.set_tiles_to_process(_tiles_to_process)
        self.PARAMS.color_match_method = kwargs.get('color_match_method', 'none'),
        
        self.CONTROLNET.name = kwargs.get('control_net_name', 'None')
        self.CONTROLNET.path = None
        self.CONTROLNET.controlnet = None
        self.CONTROLNET.low_threshold = kwargs.get('low_threshold', None)
        self.CONTROLNET.high_threshold = kwargs.get('high_threshold', None)
        self.CONTROLNET.strength = kwargs.get('strength', None)
        self.CONTROLNET.start_percent = kwargs.get('start_percent', None)
        self.CONTROLNET.end_percent = kwargs.get('end_percent', None)

        if self.CONTROLNET.name != "None":
            self.CONTROLNET.path = folder_paths.get_full_path("controlnet", self.CONTROLNET.name)
            self.CONTROLNET.controlnet = comfy.controlnet.load_controlnet(self.CONTROLNET.path)   
            
    @classmethod
    def set_tiles_to_process(self, tiles_to_process=''):

        max_tiles = len(self.OUTPUTS.grid_tiles_to_process)
        max = max_tiles if max_tiles > 0 else self.MAX_TILES
        
        def is_valid_index(index, max = self.MAX_TILES):
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
    def refine(self, tiles, iteration):
        
        grid_latents = []
        grid_latent_outputs = []
        total = len(tiles)
        
        for index, tile in enumerate(tiles):
            latent_image = None
            if len(self.PARAMS.tiles_to_process) == 0 or index in self.PARAMS.tiles_to_process:
                if self.KSAMPLER.tiled:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEEncodingTiled {iteration}")
                    latent_image = nodes.VAEEncodeTiled().encode(self.KSAMPLER.vae, tile.tile, self.KSAMPLER.tile_size_vae)[0]
                else:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEEncoding {iteration}")
                    latent_image = nodes.VAEEncode().encode(self.KSAMPLER.vae, tile.tile)[0]
            grid_latents.append(latent_image)
        
        for index, latent_image in enumerate(grid_latents):
            latent_output = None
            if len(self.PARAMS.tiles_to_process) == 0 or index in self.PARAMS.tiles_to_process:

                sigmas = self.KSAMPLER.sigmas
                if tiles[index].denoise != self.KSAMPLER.denoise:
                    denoise = tiles[index].denoise
                    sigmas = Mara_McBoaty_Configurator_v6._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, tiles[index].denoise, self.KSAMPLER.scheduler, self.KSAMPLER.model_type)
                else:
                    denoise = self.KSAMPLER.denoise
                    
                log(f"tile {index + 1}/{total} : {denoise} / {tiles[index].positive}", None, None, f"Node {self.INFO.id} - Denoise/ClipTextEncoding {iteration}")
                positive = nodes.CLIPTextEncode().encode(self.KSAMPLER.clip, tiles[index].positive)[0]
                negative = nodes.CLIPTextEncode().encode(self.KSAMPLER.clip, tiles[index].negative)[0]
                if self.CONTROLNET.controlnet is not None:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - Canny {iteration}")
                    canny_image = Canny().detect_edge(tiles[index].tile, self.CONTROLNET.low_threshold, self.CONTROLNET.high_threshold)[0]
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - ControlNetApply {iteration}")
                    positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive, negative, self.CONTROLNET.controlnet, canny_image, self.CONTROLNET.strength, self.CONTROLNET.start_percent, self.CONTROLNET.end_percent, self.KSAMPLER.vae )
                    
                log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - Refining {iteration}")
                latent_output = comfy_extras.nodes_custom_sampler.SamplerCustom().sample(
                    self.KSAMPLER.model, 
                    self.KSAMPLER.add_noise, 
                    self.KSAMPLER.noise_seed, 
                    tiles[index].cfg, 
                    positive,
                    negative,
                    self.KSAMPLER.sampler, 
                    sigmas, 
                    latent_image
                )[0]
            grid_latent_outputs.append(latent_output)

        for index, latent_output in enumerate(grid_latent_outputs):            
            output = tiles[index].tile
            if len(self.PARAMS.tiles_to_process) == 0 or index in self.PARAMS.tiles_to_process:
                if self.KSAMPLER.tiled:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEDecodingTiled {iteration}")
                    output = (nodes.VAEDecodeTiled().decode(self.KSAMPLER.vae, latent_output, self.KSAMPLER.tile_size_vae, 0)[0].unsqueeze(0))[0]
                else:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEDecoding {iteration}")
                    output = (nodes.VAEDecode().decode(self.KSAMPLER.vae, latent_output)[0].unsqueeze(0))[0]            
            tiles[index].tile = output

        return tiles

class Mara_McBoaty_TilePrompter_v6(Mara_Common_v1):

    NAME = get_name('🐰 McBoaty Tile Prompter - v6')

    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "pipe_prompty": ("MC_PROMPTY_PIPE", {"label": "McPrompty Pipe" }),
                "tiles_to_process": ("STRING", { "label": "Tile to process", "default": ""}),
                "positive": ("STRING", { "label": "Positive (Prompt)", "multiline": True, "default": self.TILE_ATTRIBUTES.positive }),
                "negative": ("STRING", { "label": "Negative (Prompt)", "multiline": True, "default": self.TILE_ATTRIBUTES.negative }),
                "cfg": ("FLOAT", { "label": "CFG", "default": self.TILE_ATTRIBUTES.cfg, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "denoise": ("FLOAT", { "label": "Denoise", "default": self.TILE_ATTRIBUTES.denoise, "min": 0.0, "max": 1.0, "step": 0.01}),
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
        
        start_time = time.time()

        tiles = kwargs.get('pipe_prompty', ([],))[0]
                
        id = kwargs.get('id', None)
        
        log("McBoaty (PromptEditor) is starting to do its magic", None, None, f"Node {id}")
        
        tile_attributes = copy.deepcopy(self.TILE_ATTRIBUTES)

        attributes = {
            'positive': kwargs.get('positive', tile_attributes.positive),
            'negative': kwargs.get('negative', tile_attributes.negative),
            'cfg': round(kwargs.get('cfg', tile_attributes.cfg), 2),
            'denoise': round(kwargs.get('denoise', tile_attributes.denoise), 2)
        }
        tiles_to_process = self.parse_tiles_to_process(kwargs.get('tiles_to_process', ""), len(tiles))
        
        if not tiles_to_process:  # This works for empty lists/arrays
            tiles_to_process = list(range(1, len(tiles) + 1))

        for id in tiles_to_process:
            index = id - 1
            for attr, value in attributes.items():
                if value != getattr(tile_attributes, attr) and value != getattr(tiles[index], attr) and value != '':
                    setattr(tiles[index], attr, value)
                    
        log("McBoaty (PromptEditor) is done with its magic", None, None, f"Node {id}")
        
        return (
            (
                tiles,
            ),
        )

                
class Mara_McBoaty_v6(Mara_McBoaty_Configurator_v6, Mara_McBoaty_Refiner_v6):

    NAME = get_name('🐰 McBoaty - v6')

    @classmethod
    def INPUT_TYPES(self):
        upscaler_inputs = Mara_McBoaty_Configurator_v6.INPUT_TYPES()
        refiner_inputs = Mara_McBoaty_Refiner_v6.INPUT_TYPES()
        
        # Merge and deduplicate inputs
        combined_inputs = {**upscaler_inputs, **refiner_inputs}
        combined_inputs['required'] = {**upscaler_inputs['required'], **refiner_inputs['required']}
        combined_inputs['optional'] = {**upscaler_inputs.get('optional', {}), **refiner_inputs.get('optional', {})}
        combined_inputs['hidden'] = {"id":"UNIQUE_ID",}
        
        combined_inputs['optional'].pop('pipe_prompty', None)        
        combined_inputs['required'].pop('tiles_to_process', None)
        
        return combined_inputs

    RETURN_TYPES = Mara_McBoaty_Refiner_v6.RETURN_TYPES
    
    RETURN_NAMES = Mara_McBoaty_Refiner_v6.RETURN_NAMES
    
    OUTPUT_IS_LIST = (False,) * len(RETURN_TYPES)
    
    OUTPUT_NODE = Mara_McBoaty_Refiner_v6.OUTPUT_NODE
    CATEGORY = get_category('Upscaling/v6')
    DESCRIPTION = "An \"UPSCALER REFINER\" Node"
    FUNCTION = "fn"

    @classmethod
    def fn(self, **kwargs):
        
        start_time = time.time()
        
        self.INFO.id = kwargs.get('id', None)
        
        # Upscaling phase
        upscaler_pipe, _, upscaler_info = Mara_McBoaty_Configurator_v6.fn(**kwargs)

        # Update kwargs with upscaler results for refiner
        kwargs.update({
            'pipe': upscaler_pipe,
        })

        # Refining phase
        mc_boaty_pipe, mc_boaty_pipe_prompty, tiles, refiner_info = Mara_McBoaty_Refiner_v6.fn(**kwargs)

        end_time = time.time()
        total_time = int(end_time - start_time)

        # Combine info from both phases
        combined_info = self._combine_info(upscaler_info, refiner_info, total_time)

        return (
            mc_boaty_pipe,
            mc_boaty_pipe_prompty,            
            tiles, 
            combined_info,
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