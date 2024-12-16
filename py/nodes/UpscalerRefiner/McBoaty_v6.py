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
from ollama import Client
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

# from ...vendor.ComfyUI_Florence2.nodes import Florence2Run
from ...vendor.ComfyUI_essentials.image import ImageTile, ImageUntile
from ...vendor.comfyui_ollama.CompfyuiOllama import Mara_OllamaVision_v1

@PromptServer.instance.routes.post("/marascott/McBoaty_v6/get_prompts")
async def get_prompts_endpoint(request):
    data = await request.json()

    tiles = data.get("tiles")
    client = Client(host=url)

    models = client.list().get('models', [])

    try:
        models = [model['model'] for model in models]
        return web.json_response(models)
    except Exception as e:
        models = [model['name'] for model in models]
        return web.json_response(models)


class Mara_Common_v1():

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
        strength = 0.76,
        start_percent = 0.000,
        end_percent = 1.000,
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
    
    NAME = "Image to tiles"
    SHORTCUT = "m"
    
    CONTROLNETS = folder_paths.get_filename_list("controlnet")
    CONTROLNET_CANNY_ONLY = ["None"]+[controlnet_name for controlnet_name in CONTROLNETS if controlnet_name is not None and ('canny' in controlnet_name.lower() or 'union' in controlnet_name.lower())]

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

                "control_net_name": (self.CONTROLNET_CANNY_ONLY , { "label": "ControlNet (Canny only)", "default": "None" }),
                "low_threshold": ("FLOAT", {"label": "Low Threshold (Canny)", "default": 0.6, "min": 0.01, "max": 0.99, "step": 0.01}),
                "high_threshold": ("FLOAT", {"label": "High Threshold (Canny)", "default": 0.6, "min": 0.01, "max": 0.99, "step": 0.01}),
                "strength": ("FLOAT", {"label": "Strength (ControlNet)", "default": 0.76, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"label": "Start % (ControlNet)", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"label": "End % (ControlNet)", "default": 0.76, "min": 0.0, "max": 1.0, "step": 0.001}),

            },
            "optional": {
            }
        }

    RETURN_TYPES = (
        "MC_BOATY_PIPE",
        "IMAGE",
        "IMAGE",
    )
    
    RETURN_NAMES = (
        "McBoayty Pipe",
        "tiles",
        "tiles - canny",
    )
    
    OUTPUT_IS_LIST = (
        False,
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

        self.OUTPUTS.tiles, self.PARAMS.tile_w, self.PARAMS.tile_h, self.PARAMS.overlap_x,  self.PARAMS.overlap_y, self.PARAMS.rows_qty, self.PARAMS.cols_qty = self.get_tiles(image=self.OUTPUTS.upscaled_image)
        
        tiles = []
        total = len(self.OUTPUTS.tiles)
        for index, tile in enumerate(self.OUTPUTS.tiles):
            _tile = copy.deepcopy(self.TILE_ATTRIBUTES)
            _tile.id = index + 1
            _tile.tile = tile.unsqueeze(0)
            _tile.canny = torch.zeros((1, _tile.tile.shape[1], _tile.tile.shape[2], 3), dtype=torch.float16)
            _tile.strength = self.CONTROLNET.strength
            _tile.start_percent = self.CONTROLNET.start_percent
            _tile.end_percent = self.CONTROLNET.end_percent
            if self.CONTROLNET.controlnet is not None:
                log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - Canny")
                _tile.canny = Canny().detect_edge(_tile.tile, self.CONTROLNET.low_threshold, self.CONTROLNET.high_threshold)[0]

            tiles.append(_tile)
            
        self.KSAMPLER.tiles = tiles

        end_time = time.time()        
        self.INFO.execution_time = int(end_time - start_time)
        
        mc_boaty_pipe = self.set_mc_boaty_pipe()
        
        self.OUTPUTS.tiles = [t.tile for t in self.KSAMPLER.tiles]
        self.OUTPUTS.cannies = [t.canny for t in self.KSAMPLER.tiles]
        
        return (
            mc_boaty_pipe,
            torch.cat(self.OUTPUTS.tiles, dim=0),
            torch.cat(self.OUTPUTS.cannies, dim=0),
        )

    @classmethod
    def init(self, **kwargs):
        
        self.INFO.id = kwargs.get('id', None)
        self.INPUTS.image = kwargs.get('image', None)

        if self.INPUTS.image is None:
            raise ValueError(f"{self.NAME} id {self.INFO.id}: No image provided")
        if not isinstance(self.INPUTS.image, torch.Tensor):
            raise ValueError(f"{self.NAME} id {self.INFO.id}: Image provided is not a Tensor")

        self.OUTPUTS.image = self.INPUTS.image
        self.OUTPUTS.tiles = self.INPUTS.image

        self.PARAMS.upscale_model_name = kwargs.get('upscale_model', 'None')
        self.PARAMS.upscale_model = None
        self.PARAMS.upscale_model_scale = 1
        self.PARAMS.tile_size = kwargs.get('tile_size', None)
        self.PARAMS.overlap = 0.10
        self.PARAMS.tile_w = 512
        self.PARAMS.tile_h = 512
        self.PARAMS.overlap_x = 0 
        self.PARAMS.overlap_y = 0
        self.PARAMS.rows_qty = 1
        self.PARAMS.cols_qty = 1
        
        if self.PARAMS.upscale_model_name != 'None':
            self.PARAMS.upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader().load_model(self.PARAMS.upscale_model_name)[0]
            self.PARAMS.upscale_model_scale = self.PARAMS.upscale_model.scale

        self.CONTROLNET.name = kwargs.get('control_net_name', 'None')
        self.CONTROLNET.low_threshold = kwargs.get('low_threshold', None)
        self.CONTROLNET.high_threshold = kwargs.get('high_threshold', None)
        self.CONTROLNET.strength = kwargs.get('strength', None)
        self.CONTROLNET.start_percent = kwargs.get('start_percent', None)
        self.CONTROLNET.end_percent = kwargs.get('end_percent', None)

        self.CONTROLNET.path = None
        self.CONTROLNET.controlnet = None
        if self.CONTROLNET.name != "None":
            self.CONTROLNET.path = folder_paths.get_full_path("controlnet", self.CONTROLNET.name)
            self.CONTROLNET.controlnet = comfy.controlnet.load_controlnet(self.CONTROLNET.path)   
    
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
        
        # return grid_images, grid_specs
        tiles, tile_w, tile_h, overlap_x, overlap_y = ImageTile().execute(image, rows_qty, cols_qty, self.PARAMS.overlap, 0, 0)

        return (tiles, tile_w, tile_h, overlap_x, overlap_y, rows_qty, cols_qty,)
        
        
class Mara_Untiler_v1(Mara_Common_v1):
    
    NAME = "Tiles to Image"
    SHORTCUT = "m"
    
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
        
        self.OUTPUTS.tiles = [t.new_tile for t in self.KSAMPLER.tiles]
        self.OUTPUTS.tiles = torch.cat(self.OUTPUTS.tiles, dim=0)        
        log(self.OUTPUTS.tiles.shape)
            
        self.OUTPUTS.image = ImageUntile().execute(
            self.OUTPUTS.tiles, 
            self.PARAMS.overlap_x, 
            self.PARAMS.overlap_y, 
            self.PARAMS.rows_qty, 
            self.PARAMS.cols_qty
        )[0]
        self.OUTPUTS.image = comfy_extras.nodes_images.ImageCrop().crop(
            self.OUTPUTS.image, 
            (self.INPUTS.image.shape[2] * self.PARAMS.upscale_model_scale), 
            (self.INPUTS.image.shape[1] * self.PARAMS.upscale_model_scale), 
            0, 
            0
        )[0]
        
        if not (self.PARAMS.upscale_size_ref == self.UPSCALE_SIZE_REF[0] and self.PARAMS.upscale_size == 1.00):
            image_ref = self.OUTPUTS.image
            if self.PARAMS.upscale_size_ref != self.UPSCALE_SIZE_REF[0]:
                image_ref = self.INPUTS.image
            self.OUTPUTS.image = nodes.ImageScale().upscale(self.OUTPUTS.image, self.PARAMS.upscale_method, int(image_ref.shape[2] * self.PARAMS.upscale_size), int(image_ref.shape[1] * self.PARAMS.upscale_size), False)[0]

        output_latent = comfy_extras.nodes_custom_sampler.SamplerCustom().sample(
            self.KSAMPLER.model, 
            self.KSAMPLER.add_noise, 
            self.KSAMPLER.noise_seed, 
            self.KSAMPLER.cfg, 
            nodes.CLIPTextEncode().encode(self.KSAMPLER.clip, self.KSAMPLER.positive)[0],
            nodes.CLIPTextEncode().encode(self.KSAMPLER.clip, self.KSAMPLER.negative)[0],
            self.KSAMPLER.sampler, 
            Mara_McBoaty_Configurator_v6._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, 0.10, self.KSAMPLER.scheduler, self.KSAMPLER.model_type), 
            nodes.VAEEncodeTiled().encode(self.KSAMPLER.vae, self.OUTPUTS.image, self.KSAMPLER.tile_size_vae)[0]
        )[0]
        self.OUTPUTS.image = nodes.VAEDecodeTiled().decode(self.KSAMPLER.vae, output_latent, self.KSAMPLER.tile_size_vae, int(self.KSAMPLER.tile_size_vae * self.PARAMS.overlap) )[0]

        if self.PARAMS.color_match_method != 'none':
            self.OUTPUTS.image = ColorMatch().colormatch(self.INPUTS.image, self.OUTPUTS.image, self.PARAMS.color_match_method)[0]

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

    NAME = "McBoaty Configurator"
    SHORTCUT = "m"
    
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
                # "Florence2": ("FL2MODEL", { "label": "Florence2" }),
                # "Llm_party": ("CUSTOM", { "label": "LLM Model" }),
                "vlm_query": ("STRING", {
                    "multiline": True,
                    "default": "generate a single paragraph prompt of max 77 token to match the red rectangle area. do not comment."
                }),
                "ollama_url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434"
                }),
                "vlm_model": ((), { "default": "llama3.2-vision:latest"}),
                "ollama_keep_alive": ("INT", {"default": 5, "min": -1, "max": 60, "step": 1}),
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
        
        self.KSAMPLER.positive = self.INPUTS.positive
        self.KSAMPLER.negative = self.INPUTS.negative
                
        if self.KSAMPLER.positive == "" and self.LLM.vision_model is not None:
            self.KSAMPLER.positive = Mara_OllamaVision_v1().ollama_vision(self.INPUTS.image, query=self.LLM.vision_query, debug="disable", url=self.LLM.ollama_url, model=self.LLM.vision_model, seed=self.KSAMPLER.noise_seed, keep_alive=self.LLM.ollama_keep_alive , format="text")[0]
        log(self.KSAMPLER.positive)
        
        for tile in self.KSAMPLER.tiles:
            tile.positive = self.KSAMPLER.positive
            if self.KSAMPLER.positive == "" and self.LLM.vision_model is not None:
                tile.positive = Mara_OllamaVision_v1().ollama_vision(tile.tile, query=self.LLM.vision_query, debug="disable", url=self.LLM.ollama_url, model=self.LLM.vision_model, seed=self.KSAMPLER.noise_seed, keep_alive=self.LLM.ollama_keep_alive , format="text")[0]
            log(tile.positive)
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

        self.INPUTS.positive = kwargs.get('positive', '')
        self.INPUTS.negative = kwargs.get('negative', '')
                
        self.LLM.vision_query = kwargs.get("vlm_query", None)
        self.LLM.ollama_url = kwargs.get("ollama_url", None)
        self.LLM.vision_model = kwargs.get("vlm_model", None)
        self.LLM.ollama_keep_alive = kwargs.get("ollama_keep_alive", None)        
        
        self.PARAMS.tile_prompting_active = kwargs.get('tile_prompting_active', False)        

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

        # TODO : make the feather_mask proportional to tile size ?
        # self.PARAMS.feather_mask = self.PARAMS.tile_size // 16
        self.PARAMS.feather_mask = 0

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
    
    # @classmethod
    # def upscale(self, image):
        
    #     rows_qty_float = (image.shape[1] * self.PARAMS.upscale_model_scale) / self.PARAMS.tile_size
    #     cols_qty_float = (image.shape[2] * self.PARAMS.upscale_model_scale) / self.PARAMS.tile_size
    #     rows_qty = math.ceil(rows_qty_float)
    #     cols_qty = math.ceil(cols_qty_float)

    #     tiles_qty = rows_qty * cols_qty        
    #     if tiles_qty > self.MAX_TILES :
    #         msg = get_log(f"\n\n--------------------\n\n!!! Number of tiles is higher than {self.MAX_TILES} ({tiles_qty} for {self.PARAMS.cols_qty} cols and {self.PARAMS.rows_qty} rows)!!!\n\nPlease consider increasing your tile and feather sizes\n\n--------------------\n", "BLUE", "YELLOW", f"Node {self.INFO.id} - Mara_McBoaty_Configurator_v6")
    #         raise ValueError(msg)

    #     upscaled_image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(self.PARAMS.upscale_model, image)[0]

    #     self.PARAMS.rows_qty = rows_qty
    #     self.PARAMS.cols_qty = cols_qty
        
    #     # grid_specs = MS_Image().get_dynamic_grid_specs(upscaled_image.shape[2], upscaled_image.shape[1], rows_qty, cols_qty, self.PARAMS.feather_mask)[0]
    #     grid_specs = MS_Image().get_tiled_grid_specs(upscaled_image, self.PARAMS.tile_size, self.PARAMS.rows_qty, self.PARAMS.cols_qty, self.PARAMS.feather_mask)[0]
    #     grid_images = MS_Image().get_grid_images(upscaled_image, grid_specs)
        
    #     grid_prompts = []
    #     llm = MS_Llm(self.LLM.vision_model, self.LLM.model)
    #     prompt_context = llm.vision_llm.generate_prompt(image)
    #     total = len(grid_images)
    #     for index, grid_image in enumerate(grid_images):
    #         prompt_tile = prompt_context
    #         if self.PARAMS.tile_prompting_active:
    #             log(f"tile {index + 1}/{total} - [tile prompt]", None, None, f"Node {self.INFO.id} - Prompting")
    #             prompt_tile = llm.generate_tile_prompt(grid_image, prompt_context, self.KSAMPLER.noise_seed)
    #         log(f"tile {index + 1}/{total} - [tile prompt] {prompt_tile}", None, None, f"Node {self.INFO.id} - Prompting")
    #         grid_prompts.append(prompt_tile)
                            
    #     return grid_specs, grid_images, grid_prompts

class Mara_McBoaty_Refiner_v6(Mara_Common_v1):
    
    NAME = "McBoaty Refiner"
    SHORTCUT = "m"

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
        "IMAGE",
        "STRING"
    )
    
    RETURN_NAMES = (
        "McBoaty Pipe", 
        "McPrompty Pipe",
        "tiles", 
        "tiles - cannies", 
        "info", 
    )
    
    OUTPUT_IS_LIST = (
        False,
        False,
        False,
        False,
        False,
    )
    
    
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
        self.KSAMPLER.tiles = self.override_tiles(self.KSAMPLER.tiles, tiles)
        self.KSAMPLER.tiles = self.refine(self.KSAMPLER.tiles)
        end_time = time.time()

        output_info = self._get_info(
            int(end_time - start_time)
        )

        mc_boaty_pipe = self.set_mc_boaty_pipe()
        
        self.OUTPUTS.tiles = [t.new_tile for t in self.KSAMPLER.tiles]
        self.OUTPUTS.cannies = [t.canny for t in self.KSAMPLER.tiles]

        log("McBoaty (Refiner) is done with its magic", None, None, f"Node {self.INFO.id}")

        return (
            mc_boaty_pipe,
            (
                self.KSAMPLER.tiles,
            ),            
            torch.cat(self.OUTPUTS.tiles, dim=0),
            torch.cat(self.OUTPUTS.cannies, dim=0),
            output_info, 
        )
        
    @classmethod
    def init(self, **kwargs):
        
        pipe = kwargs.get('pipe', (SimpleNamespace(),) * len(self.PIPE_ATTRIBUTES))
        self.set_pipe_values(pipe)

        self.INFO.id = kwargs.get('id', None)        

        _tiles_to_process = kwargs.get('tiles_to_process', '')
        self.PARAMS.tiles_to_process = self.set_tiles_to_process(_tiles_to_process)
        self.PARAMS.color_match_method = kwargs.get('color_match_method', 'none')
                    
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
    def refine(self, tiles):
        
        total = len(tiles)
        for index, tile in enumerate(tiles):
            _tile = tile.tile
            _tile = _tile.squeeze(0).permute(2, 0, 1)
            _, h, w = _tile.shape
            dim_max = max(h, w)
            if dim_max < self.PARAMS.tile_size:
                dim_max = self.PARAMS.tile_size
            else:
                _tile, _, _, self.INFO.is_image_divisible_by_8 = MS_Image().format_2_divby8(image=tile.tile)
                _tile = _tile.squeeze(0).permute(2, 0, 1)
                _, h, w = _tile.shape
                dim_max = max(h, w)

            new_tile, tile_padding = MS_Image.pad_to_square(_tile, dim_max)
            new_tile = new_tile.permute(1, 2, 0).unsqueeze(0)
            if len(self.PARAMS.tiles_to_process) == 0 or index in self.PARAMS.tiles_to_process:
                if self.KSAMPLER.tiled:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEEncodingTiled")
                    tile.latent = nodes.VAEEncodeTiled().encode(self.KSAMPLER.vae, new_tile, self.KSAMPLER.tile_size_vae)[0]
                else:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEEncoding")
                    tile.latent = nodes.VAEEncode().encode(self.KSAMPLER.vae, new_tile)[0]
        
                sigmas = self.KSAMPLER.sigmas
                if tile.denoise != self.KSAMPLER.denoise:
                    denoise = tile.denoise
                    sigmas = Mara_McBoaty_Configurator_v6._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, tile.denoise, self.KSAMPLER.scheduler, self.KSAMPLER.model_type)
                else:
                    denoise = self.KSAMPLER.denoise
                    
                log(f"tile {index + 1}/{total} : {denoise} / {tile.positive}", None, None, f"Node {self.INFO.id} - Denoise/ClipTextEncoding")
                positive = nodes.CLIPTextEncode().encode(self.KSAMPLER.clip, tile.positive)[0]
                negative = nodes.CLIPTextEncode().encode(self.KSAMPLER.clip, tile.negative)[0]
                tile.controlnet = True
                if self.CONTROLNET.controlnet is not None and tile.controlnet:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - ControlNetApply")
                    tile.canny = Canny().detect_edge(new_tile, self.CONTROLNET.low_threshold, self.CONTROLNET.high_threshold)[0]
                    positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive, negative, self.CONTROLNET.controlnet, tile.canny, tile.strength, tile.start_percent, tile.end_percent, self.KSAMPLER.vae )
                    
                log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - Refining")
                _latent = tile.latent
                _latent = comfy_extras.nodes_custom_sampler.SamplerCustom().sample(
                    self.KSAMPLER.model, 
                    self.KSAMPLER.add_noise, 
                    self.KSAMPLER.noise_seed, 
                    tile.cfg, 
                    positive,
                    negative,
                    self.KSAMPLER.sampler, 
                    sigmas, 
                    _latent
                )[0]
                tile.latent = _latent

                if self.KSAMPLER.tiled:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEDecodingTiled")
                    new_tile = nodes.VAEDecodeTiled().decode(self.KSAMPLER.vae, tile.latent, self.KSAMPLER.tile_size_vae, int(self.KSAMPLER.tile_size_vae * self.PARAMS.overlap))[0]
                else:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {self.INFO.id} - VAEDecoding")
                    new_tile = nodes.VAEDecode().decode(self.KSAMPLER.vae, tile.latent)[0]
                _new_tile = new_tile
                new_tile = new_tile.squeeze(0).permute(2, 0, 1)
                new_tile = MS_Image.crop_to_original(new_tile, _tile.shape, tile_padding)
                tile.new_tile = new_tile.permute(1, 2, 0).unsqueeze(0)                
                log((_tile.shape, dim_max, tile_padding, _new_tile.shape, new_tile.shape))
        
        return tiles

class Mara_McBoaty_TilePrompter_v6(Mara_Common_v1):

    NAME = "McBoaty Tile Prompter"
    SHORTCUT = "m"

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
                "strength": ("FLOAT", {"label": "Strength (ControlNet)", "default": 0.76, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"label": "Start % (ControlNet)", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"label": "End % (ControlNet)", "default": 0.76, "min": 0.0, "max": 1.0, "step": 0.001}),
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
            'denoise': round(kwargs.get('denoise', tile_attributes.denoise), 2),
            'strength': round(kwargs.get('strength', tile_attributes.strength), 2),
            'start_percent': round(kwargs.get('start_percent', tile_attributes.start_percent), 3),
            'end_percent': round(kwargs.get('end_percent', tile_attributes.end_percent), 3)
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

    NAME = "McBoaty"
    SHORTCUT = "m"

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
    
    OUTPUT_IS_LIST = Mara_McBoaty_Refiner_v6.OUTPUT_IS_LIST
    
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
        mc_boaty_pipe, mc_boaty_pipe_prompty, tiles, cannies, refiner_info = Mara_McBoaty_Refiner_v6.fn(**kwargs)

        end_time = time.time()
        total_time = int(end_time - start_time)

        # Combine info from both phases
        combined_info = self._combine_info(upscaler_info, refiner_info, total_time)
        
        return (
            mc_boaty_pipe,
            mc_boaty_pipe_prompty,            
            tiles, 
            cannies,
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