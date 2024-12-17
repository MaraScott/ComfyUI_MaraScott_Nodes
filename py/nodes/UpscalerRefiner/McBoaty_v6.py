#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Upscaler Refiner Node.  Upscale and Refine a picture by 2 using a 9 Square Grid to upscale and refine the visual in 9 sequences
#
###

# import os
# import sys
import time
import copy
# import glob
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
# from ollama import Client
import folder_paths

# from PIL import Image
# import numpy as np

# from .... import root_dir, __MARASCOTT_TEMP__
from ...utils.constants import get_name, get_category

from ...utils.version import VERSION
from ...inc.lib.image import MS_Image_v2 as MS_Image
from ...vendor.ComfyUI_KJNodes.nodes.image_nodes import ColorMatch as ColorMatch
# from ...inc.lib.llm import MS_Llm
# from ...inc.lib.cache import MS_Cache

from ...utils.log import log, get_log, COLORS

# from ...vendor.ComfyUI_Florence2.nodes import Florence2Run
from ...vendor.ComfyUI_essentials.image import ImageTile, ImageUntile
from ...vendor.comfyui_ollama.CompfyuiOllama import Mara_OllamaVision_v1

class Store:
    
    empty_data = {"positive": "", "negative": ""}
    
    def __init__(self, initial_state=None):
        self._state = initial_state or {}
        self._subscribers = []  # List of subscribers for change notification
        
    def has_index(cls, index_id):
        index_id = str(index_id)
        """Check if an index exists in the store."""
        return index_id in cls._state

    def add_item(self, index_id, index_prompt, value):
        index_id = str(index_id)
        index_prompt = str(index_prompt)        
        """Add or update an item under a specific index and prompt."""
        if index_id not in self._state:
            self._state[index_id] = {}  # Initialize if index_id doesn't exist
        self._state[index_id][index_prompt] = value

    def get_item(self, index_id, index_prompt = None):
        index_id = str(index_id)
        index_prompt = str(index_prompt)
        """Get an item by index and optional prompt."""
        data = self._state.get(index_id, { "all": self.empty_data })
        if index_prompt:
            data = data.get(index_prompt, self.empty_data)
        return data

    def get_state(self):
        """Get the current state."""
        return self._state

    def set_state(self, new_state):
        """Update the state and notify subscribers."""
        self._state.update(new_state)
        self._notify_subscribers()

    def subscribe(self, callback):
        """Add a subscriber callback to notify on state changes."""
        self._subscribers.append(callback)

    def _notify_subscribers(self):
        """Notify all subscribers about the state change."""
        for callback in self._subscribers:
            callback(self._state)


# Example Usage
store = Store(initial_state={})

def on_state_change(new_state):
    log(new_state, None, None, "Prompts")

store.subscribe(on_state_change)

@PromptServer.instance.routes.post("/marascott/McBoaty_v6/get_prompts")
async def get_prompts_endpoint(request):
    data = await request.json()

    tiles = data.get("tiles", "all")
    id = data.get("parentId", None)
    tiles = Mara_Common_v1.parse_tiles_to_process(tiles)
    tiles = ["all"] if len(tiles) == 0 else tiles
    prompts = Store.empty_data
    if id is not None and len(tiles) > 0:
        prompts = store.get_item(id, tiles[0])

    try:
        return web.json_response(prompts)
    except Exception as e:
        prompts = [{"positive": "", "negative": ""}]
        return web.json_response(prompts)


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
    
    @staticmethod
    def init(local_PIPE=SimpleNamespace()):
        # Dynamically create attributes from PIPE_ATTRIBUTES
        for attr in copy.deepcopy(Mara_Common_v1.PIPE_ATTRIBUTES):
            if not hasattr(local_PIPE, attr):  # Only set the attribute if it doesn't already exist
                setattr(local_PIPE, attr, SimpleNamespace())
        return local_PIPE

    @staticmethod
    def set_pipe_values(local_PIPE, pipe):
        for name, value in zip(copy.deepcopy(Mara_Common_v1.PIPE_ATTRIBUTES), pipe):
            setattr(local_PIPE, name, value)
        return local_PIPE

    @staticmethod
    def set_mc_boaty_pipe(local_PIPE):
        local_PIPE = copy.deepcopy(local_PIPE)
        return tuple(getattr(local_PIPE, attr, None) for attr in copy.deepcopy(Mara_Common_v1.PIPE_ATTRIBUTES))
    
    @staticmethod
    def parse_tiles_to_process(tiles_to_process = "", MAX_TILES = 16384):
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

    @staticmethod
    def override_tiles(local_PIPE, tiles, new_tiles):
        
        tiles = copy.deepcopy(tiles)
        
        for index, tile in enumerate(tiles):
            if index >= len(new_tiles):
                continue  # Skip if the index doesn't exist in the `tiles` list
            
            if len(local_PIPE.PARAMS.tiles_to_process) == 0 or index in local_PIPE.PARAMS.tiles_to_process:
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
                            # Compare non-tensor attributes
                            if isinstance(override_value, dict):
                                # If override_value is a dict, manually compare each key-value pair
                                for k, v in override_value.items():
                                    if k in tile_value and not torch.equal(tile_value[k], v):
                                        tile_value[k] = v
                            elif tile_value != override_value:
                                setattr(tiles[index], attr, override_value)
        return tiles

    
class Mara_Tiler_v1():
    
    NAME = "Image to tiles"
    SHORTCUT = "m"
    
    CONTROLNETS = folder_paths.get_filename_list("controlnet")
    CONTROLNET_CANNY_ONLY = ["None"]+[controlnet_name for controlnet_name in CONTROLNETS if controlnet_name is not None and ('canny' in controlnet_name.lower() or 'union' in controlnet_name.lower())]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "image": ("IMAGE", {"label": "Image" }),
                "upscale_model": (["None"]+folder_paths.get_filename_list("upscale_models"), { "label": "Upscale Model" }),
                "tile_size": ("INT", { "label": "Tile Size", "default": 512, "min": 320, "max": 4096, "step": 64}),

                "control_net_name": (cls.CONTROLNET_CANNY_ONLY , { "label": "ControlNet (Canny only)", "default": "None" }),
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

        local_PIPE = self.init(**kwargs)
        
        log("McBoaty (Tiler) is starting to slicing the image", None, None, f"Node {local_PIPE.INFO.id}")
        
        local_PIPE.OUTPUTS.image, _, _, local_PIPE.INFO.is_image_divisible_by_8 = MS_Image().format_2_divby8(image=local_PIPE.INPUTS.image)
        local_PIPE.OUTPUTS.upscaled_image = local_PIPE.OUTPUTS.image 
        if local_PIPE.PARAMS.upscale_model is not None:
            local_PIPE.OUTPUTS.upscaled_image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(local_PIPE.PARAMS.upscale_model, local_PIPE.OUTPUTS.upscaled_image)[0]
        local_PIPE.INFO.image_width = local_PIPE.OUTPUTS.upscaled_image.shape[1]
        local_PIPE.INFO.image_height = local_PIPE.OUTPUTS.upscaled_image.shape[2]

        local_PIPE.OUTPUTS.tiles, local_PIPE.PARAMS.tile_w, local_PIPE.PARAMS.tile_h, local_PIPE.PARAMS.overlap_x,  local_PIPE.PARAMS.overlap_y, local_PIPE.PARAMS.rows_qty, local_PIPE.PARAMS.cols_qty = self.get_tiles(pipe=local_PIPE, image=local_PIPE.OUTPUTS.upscaled_image)
        
        tiles = []
        total = len(local_PIPE.OUTPUTS.tiles)
        for index, tile in enumerate(local_PIPE.OUTPUTS.tiles):
            _tile = copy.deepcopy(Mara_Common_v1.TILE_ATTRIBUTES)
            _tile.id = index + 1
            _tile.tile = tile.unsqueeze(0)
            _tile.canny = torch.zeros((1, _tile.tile.shape[1], _tile.tile.shape[2], 3), dtype=torch.float16)
            _tile.strength = local_PIPE.CONTROLNET.strength
            _tile.start_percent = local_PIPE.CONTROLNET.start_percent
            _tile.end_percent = local_PIPE.CONTROLNET.end_percent
            if local_PIPE.CONTROLNET.controlnet is not None:
                log(f"tile {index + 1}/{total}", None, None, f"Node {local_PIPE.INFO.id} - Canny")
                _tile.canny = Canny().detect_edge(_tile.tile, local_PIPE.CONTROLNET.low_threshold, local_PIPE.CONTROLNET.high_threshold)[0]

            tiles.append(_tile)
            
        local_PIPE.KSAMPLER.tiles = tiles

        end_time = time.time()        
        local_PIPE.INFO.execution_time = int(end_time - start_time)
        
        mc_boaty_pipe = Mara_Common_v1.set_mc_boaty_pipe(local_PIPE)
        
        tiles = [t.tile for t in local_PIPE.KSAMPLER.tiles]
        cannies = [t.canny for t in local_PIPE.KSAMPLER.tiles]
        
        return (
            mc_boaty_pipe,
            torch.cat(tiles, dim=0),
            torch.cat(cannies, dim=0),
        )

    @classmethod
    def init(self, **kwargs):
        
        local_PIPE = Mara_Common_v1().init()
        
        local_PIPE.INFO.id = kwargs.get('id', None)
        local_PIPE.INPUTS.image = kwargs.get('image', None)

        if local_PIPE.INPUTS.image is None:
            raise ValueError(f"{self.NAME} id {local_PIPE.INFO.id}: No image provided")
        if not isinstance(local_PIPE.INPUTS.image, torch.Tensor):
            raise ValueError(f"{self.NAME} id {local_PIPE.INFO.id}: Image provided is not a Tensor")

        local_PIPE.OUTPUTS.image = local_PIPE.INPUTS.image
        local_PIPE.OUTPUTS.tiles = local_PIPE.INPUTS.image

        local_PIPE.PARAMS.upscale_model_name = kwargs.get('upscale_model', 'None')
        local_PIPE.PARAMS.upscale_model = None
        local_PIPE.PARAMS.upscale_model_scale = 1
        local_PIPE.PARAMS.tile_size = kwargs.get('tile_size', None)
        local_PIPE.PARAMS.overlap = 0.10
        local_PIPE.PARAMS.tile_w = 512
        local_PIPE.PARAMS.tile_h = 512
        local_PIPE.PARAMS.overlap_x = 0 
        local_PIPE.PARAMS.overlap_y = 0
        local_PIPE.PARAMS.rows_qty = 1
        local_PIPE.PARAMS.cols_qty = 1
        
        if local_PIPE.PARAMS.upscale_model_name != 'None':
            local_PIPE.PARAMS.upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader().load_model(local_PIPE.PARAMS.upscale_model_name)[0]
            local_PIPE.PARAMS.upscale_model_scale = local_PIPE.PARAMS.upscale_model.scale

        local_PIPE.CONTROLNET.name = kwargs.get('control_net_name', 'None')
        local_PIPE.CONTROLNET.low_threshold = kwargs.get('low_threshold', None)
        local_PIPE.CONTROLNET.high_threshold = kwargs.get('high_threshold', None)
        local_PIPE.CONTROLNET.strength = kwargs.get('strength', None)
        local_PIPE.CONTROLNET.start_percent = kwargs.get('start_percent', None)
        local_PIPE.CONTROLNET.end_percent = kwargs.get('end_percent', None)

        local_PIPE.CONTROLNET.path = None
        local_PIPE.CONTROLNET.controlnet = None
        if local_PIPE.CONTROLNET.name != "None":
            local_PIPE.CONTROLNET.path = folder_paths.get_full_path("controlnet", local_PIPE.CONTROLNET.name)
            local_PIPE.CONTROLNET.controlnet = comfy.controlnet.load_controlnet(local_PIPE.CONTROLNET.path)   
            
        return local_PIPE
    
    @classmethod
    def get_tiles(self, pipe, image):
        rows_qty_float = (image.shape[1]) / pipe.PARAMS.tile_size
        cols_qty_float = (image.shape[2]) / pipe.PARAMS.tile_size
        rows_qty = math.ceil(rows_qty_float)
        cols_qty = math.ceil(cols_qty_float)

        tiles_qty = rows_qty * cols_qty        
        if tiles_qty > Mara_Common_v1.MAX_TILES :
            msg = get_log(f"\n\n--------------------\n\n!!! Number of tiles is higher than {Mara_Common_v1.MAX_TILES} ({tiles_qty} for {pipe.PARAMS.cols_qty} cols and {pipe.PARAMS.rows_qty} rows)!!!\n\nPlease consider increasing your tile and feather sizes\n\n--------------------\n", "BLUE", "YELLOW", f"Node {pipe.INFO.id} - {self.NAME}")
            raise ValueError(msg)
        
        # return grid_images, grid_specs
        tiles, tile_w, tile_h, overlap_x, overlap_y = ImageTile().execute(image, rows_qty, cols_qty, pipe.PARAMS.overlap, 0, 0)

        return (tiles, tile_w, tile_h, overlap_x, overlap_y, rows_qty, cols_qty,)
        
        
class Mara_Untiler_v1():
    
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
    def INPUT_TYPES(cls):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "pipe": ("MC_BOATY_PIPE", {"label": "McBoaty Pipe" }),
                "output_upscale_method": (cls.UPSCALE_METHODS, { "label": "Custom Output Upscale Method", "default": "bicubic"}),
                "output_size_ref": (cls.UPSCALE_SIZE_REF, { "label": "Output Size Ref", "default": "Output Image"}),
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

        local_PIPE = self.init(**kwargs)
        
        log("McBoaty (Untiler) is starting to rebuild the image", None, None, f"Node {local_PIPE.INFO.id}")
        
        local_PIPE.OUTPUTS.tiles = [t.new_tile for t in local_PIPE.KSAMPLER.tiles]
        local_PIPE.OUTPUTS.tiles = torch.cat(local_PIPE.OUTPUTS.tiles, dim=0)
            
        local_PIPE.OUTPUTS.image = ImageUntile().execute(
            local_PIPE.OUTPUTS.tiles,
            local_PIPE.PARAMS.overlap_x, 
            local_PIPE.PARAMS.overlap_y, 
            local_PIPE.PARAMS.rows_qty, 
            local_PIPE.PARAMS.cols_qty
        )[0]
        local_PIPE.OUTPUTS.image = comfy_extras.nodes_images.ImageCrop().crop(
            local_PIPE.OUTPUTS.image, 
            (local_PIPE.INPUTS.image.shape[2] * local_PIPE.PARAMS.upscale_model_scale), 
            (local_PIPE.INPUTS.image.shape[1] * local_PIPE.PARAMS.upscale_model_scale), 
            0, 
            0
        )[0]
        
        if not (local_PIPE.PARAMS.upscale_size_ref == self.UPSCALE_SIZE_REF[0] and local_PIPE.PARAMS.upscale_size == 1.00):
            image_ref = local_PIPE.OUTPUTS.image
            if local_PIPE.PARAMS.upscale_size_ref != self.UPSCALE_SIZE_REF[0]:
                image_ref = local_PIPE.INPUTS.image
            local_PIPE.OUTPUTS.image = nodes.ImageScale().upscale(local_PIPE.OUTPUTS.image, local_PIPE.PARAMS.upscale_method, int(image_ref.shape[2] * local_PIPE.PARAMS.upscale_size), int(image_ref.shape[1] * local_PIPE.PARAMS.upscale_size), False)[0]

        output_latent = comfy_extras.nodes_custom_sampler.SamplerCustom().sample(
            local_PIPE.KSAMPLER.model, 
            local_PIPE.KSAMPLER.add_noise, 
            local_PIPE.KSAMPLER.noise_seed, 
            local_PIPE.KSAMPLER.cfg, 
            nodes.CLIPTextEncode().encode(local_PIPE.KSAMPLER.clip, local_PIPE.KSAMPLER.positive)[0],
            nodes.CLIPTextEncode().encode(local_PIPE.KSAMPLER.clip, local_PIPE.KSAMPLER.negative)[0],
            local_PIPE.KSAMPLER.sampler, 
            Mara_McBoaty_Configurator_v6._get_sigmas(local_PIPE.KSAMPLER.sigmas_type, local_PIPE.KSAMPLER.model, local_PIPE.KSAMPLER.steps, 0.10, local_PIPE.KSAMPLER.scheduler, local_PIPE.KSAMPLER.model_type), 
            nodes.VAEEncodeTiled().encode(local_PIPE.KSAMPLER.vae, local_PIPE.OUTPUTS.image, local_PIPE.KSAMPLER.tile_size_vae)[0]
        )[0]
        local_PIPE.OUTPUTS.image = nodes.VAEDecodeTiled().decode(local_PIPE.KSAMPLER.vae, output_latent, local_PIPE.KSAMPLER.tile_size_vae, int(local_PIPE.KSAMPLER.tile_size_vae * local_PIPE.PARAMS.overlap) )[0]

        if local_PIPE.PARAMS.color_match_method != 'none':
            local_PIPE.OUTPUTS.image = ColorMatch().colormatch(local_PIPE.INPUTS.image, local_PIPE.OUTPUTS.image, local_PIPE.PARAMS.color_match_method)[0]

        end_time = time.time()
        local_PIPE.INFO.execution_time = int(end_time - start_time)

        return (
            copy.deepcopy(local_PIPE.OUTPUTS.image),
        )
        
    @classmethod
    def init(cls, **kwargs):

        local_PIPE = Mara_Common_v1().init()

        pipe = kwargs.get('pipe', (SimpleNamespace(),) * len(Mara_Common_v1.PIPE_ATTRIBUTES))

        id = kwargs.get('id', None)
        # log(pipe[0], None, None, f"Node {id}") TODO
        
        local_PIPE = Mara_Common_v1.set_pipe_values(local_PIPE, pipe)

        local_PIPE.INFO.id = kwargs.get('id', None)
        local_PIPE.INPUTS.tiles = kwargs.get('tiles', None)
        local_PIPE.OUTPUTS.image = local_PIPE.INPUTS.image

        local_PIPE.PARAMS.upscale_size_ref = kwargs.get('output_size_ref', False)
        local_PIPE.PARAMS.upscale_size = kwargs.get('output_size', 1.00)
        local_PIPE.PARAMS.upscale_method = kwargs.get('output_upscale_method', "lanczos")
        
        return local_PIPE
        
class Mara_McBoaty_Configurator_v6():

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
    def INPUT_TYPES(cls):
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
                "sigmas_type": (cls.SIGMAS_TYPES, { "label": "Sigmas Type" }),
                "model_type": (cls.MODEL_TYPES, { "label": "Model Type", "default": "SDXL" }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, { "label": "Sampler Name" }),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, { "label": "Basic Scheduler" }),
                "steps": ("INT", { "label": "Steps", "default": 10, "min": 1, "max": 10000}),
                "seed": ("INT", { "label": "Seed", "default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", { "label": "CFG", "default": Mara_Common_v1.TILE_ATTRIBUTES.cfg, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "denoise": ("FLOAT", { "label": "Denoise", "default": Mara_Common_v1.TILE_ATTRIBUTES.denoise, "min": 0.0, "max": 1.0, "step": 0.01}),

                "vae_encode": ("BOOLEAN", { "label": "VAE Encode type", "default": True, "label_on": "tiled", "label_off": "standard"}),
                "tile_size_vae": ("INT", { "label": "Tile Size (VAE)", "default": 512, "min": 256, "max": 4096, "step": 64}),

            },
            "optional": {
                "tiles": ("IMAGE", {"label": "Tiles" }),
                # "Florence2": ("FL2MODEL", { "label": "Florence2" }),
                # "Llm_party": ("CUSTOM", { "label": "LLM Model" }),
                "vlm_query": ("STRING", {
                    "multiline": True,
                    "default": "generate a single paragraph prompt of max 77 token to describe the image. do not comment."
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
        
        local_PIPE = self.init(**kwargs)
        
        log("McBoaty (Upscaler) is starting to do its magic", None, None, f"Node {local_PIPE.INFO.id}")
        
        local_PIPE.KSAMPLER.positive = local_PIPE.INPUTS.positive
        local_PIPE.KSAMPLER.negative = local_PIPE.INPUTS.negative
        store.add_item(local_PIPE.INFO.id, "all", {"positive": local_PIPE.KSAMPLER.positive, "negative": local_PIPE.KSAMPLER.negative})
        generate_prompt = local_PIPE.KSAMPLER.positive == "" and local_PIPE.LLM.vision_model is not None
        if generate_prompt:
            log("Initiate Image Analysis", None, None, f"Node {local_PIPE.INFO.id} - OllamaVision - Entire Image")
            local_PIPE.KSAMPLER.positive = Mara_OllamaVision_v1().ollama_vision(local_PIPE.INPUTS.image, query=local_PIPE.LLM.vision_query, debug="disable", url=local_PIPE.LLM.ollama_url, model=local_PIPE.LLM.vision_model, seed=local_PIPE.KSAMPLER.noise_seed, keep_alive=local_PIPE.LLM.ollama_keep_alive , format="text")[0]
            log(f"{local_PIPE.KSAMPLER.positive}", None, None, f"Node {local_PIPE.INFO.id} - OllamaVision - Entire Image")
            log("End Image Analysis", None, None, f"Node {local_PIPE.INFO.id} - OllamaVision - Entire Image")
            store.add_item(local_PIPE.INFO.id, "all", {"positive": local_PIPE.KSAMPLER.positive, "negative": local_PIPE.KSAMPLER.negative})
        
        for index, tile in enumerate(local_PIPE.KSAMPLER.tiles):
            tile.positive = local_PIPE.KSAMPLER.positive
            if generate_prompt:
                log(f"Initiate Tile {index} Analysis", None, None, f"Node {local_PIPE.INFO.id} - OllamaVision - Tile")
                tile.positive = Mara_OllamaVision_v1().ollama_vision(tile.tile, query=local_PIPE.LLM.vision_query, debug="disable", url=local_PIPE.LLM.ollama_url, model=local_PIPE.LLM.vision_model, seed=local_PIPE.KSAMPLER.noise_seed, keep_alive=local_PIPE.LLM.ollama_keep_alive , format="text")[0]
                log(f"Tile {index} : {tile.positive}", None, None, f"Node {local_PIPE.INFO.id} - OllamaVision - Tile")
                log(f"End Tile {index} Analysis", None, None, f"Node {local_PIPE.INFO.id} - OllamaVision - Tile")
            tile.negative = local_PIPE.KSAMPLER.negative
            tile.cfg = local_PIPE.KSAMPLER.cfg
            tile.denoise = local_PIPE.KSAMPLER.denoise
            store.add_item(local_PIPE.INFO.id, index, {"positive": local_PIPE.KSAMPLER.positive, "negative": local_PIPE.KSAMPLER.negative})
            
        end_time = time.time()

        output_info = self._get_info(
            0, # local_PIPE.INFO.image_width, 
            0, # local_PIPE.INFO.image_height, 
            True, # local_PIPE.INFO.image_divisible_by_8, 
            int(end_time - start_time)
        )
        
        mc_boaty_pipe = Mara_Common_v1.set_mc_boaty_pipe(local_PIPE)
        
        log("McBoaty (Upscaler) is done with its magic", None, None, f"Node {local_PIPE.INFO.id}")

        return (
            mc_boaty_pipe,
            (
                copy.deepcopy(local_PIPE.KSAMPLER.tiles),
            ),
            output_info
        )
        
    @classmethod
    def init(self, **kwargs):

        local_PIPE = Mara_Common_v1().init()

        pipe = kwargs.get('pipe', (SimpleNamespace(),) * len(Mara_Common_v1.PIPE_ATTRIBUTES))
        local_PIPE = Mara_Common_v1.set_pipe_values(local_PIPE, pipe)

        local_PIPE.INFO.id = kwargs.get('id', None)
        
        local_PIPE.INPUTS.tiles = kwargs.get('tiles', None)
        if local_PIPE.INPUTS.tiles is not None and not isinstance(local_PIPE.INPUTS.tiles, torch.Tensor):
            raise ValueError(f"{self.NAME} id {local_PIPE.INFO.id}: tiles provided are not Tensors")

        local_PIPE.INPUTS.positive = kwargs.get('positive', '')
        local_PIPE.INPUTS.negative = kwargs.get('negative', '')
                
        local_PIPE.LLM.vision_query = kwargs.get("vlm_query", None)
        local_PIPE.LLM.ollama_url = kwargs.get("ollama_url", None)
        local_PIPE.LLM.vision_model = kwargs.get("vlm_model", None)
        local_PIPE.LLM.ollama_keep_alive = kwargs.get("ollama_keep_alive", None)        
        
        local_PIPE.PARAMS.tile_prompting_active = kwargs.get('tile_prompting_active', False)        

        local_PIPE.KSAMPLER.tiled = kwargs.get('vae_encode', None)
        local_PIPE.KSAMPLER.tile_size_vae = kwargs.get('tile_size_vae', None)
        local_PIPE.KSAMPLER.model = kwargs.get('model', None)
        local_PIPE.KSAMPLER.clip = kwargs.get('clip', None)
        local_PIPE.KSAMPLER.vae = kwargs.get('vae', None)
        local_PIPE.KSAMPLER.noise_seed = kwargs.get('seed', None)
        local_PIPE.KSAMPLER.add_noise = True

        local_PIPE.KSAMPLER.sigmas_type = kwargs.get('sigmas_type', None)
        local_PIPE.KSAMPLER.model_type = kwargs.get('model_type', None)
        local_PIPE.KSAMPLER.tile_size_sampler = self.MODEL_TYPE_SIZES[local_PIPE.KSAMPLER.model_type]
        local_PIPE.KSAMPLER.sampler_name = kwargs.get('sampler_name', None)
        local_PIPE.KSAMPLER.sampler = comfy_extras.nodes_custom_sampler.KSamplerSelect().get_sampler(local_PIPE.KSAMPLER.sampler_name)[0]
        local_PIPE.KSAMPLER.scheduler = kwargs.get('basic_scheduler', None)
        local_PIPE.KSAMPLER.steps = kwargs.get('steps', None)
        local_PIPE.KSAMPLER.positive = kwargs.get('positive', '')
        local_PIPE.KSAMPLER.negative = kwargs.get('negative', '')
        local_PIPE.KSAMPLER.cfg = kwargs.get('cfg', Mara_Common_v1.TILE_ATTRIBUTES.cfg)
        local_PIPE.KSAMPLER.denoise = kwargs.get('denoise', Mara_Common_v1.TILE_ATTRIBUTES.denoise)
        
        local_PIPE.KSAMPLER.sigmas = self._get_sigmas(local_PIPE.KSAMPLER.sigmas_type, local_PIPE.KSAMPLER.model, local_PIPE.KSAMPLER.steps, local_PIPE.KSAMPLER.denoise, local_PIPE.KSAMPLER.scheduler, local_PIPE.KSAMPLER.model_type)
        # local_PIPE.KSAMPLER.outpaint_sigmas = self._get_sigmas(local_PIPE.KSAMPLER.sigmas_type, local_PIPE.KSAMPLER.model, local_PIPE.KSAMPLER.steps, 1, local_PIPE.KSAMPLER.scheduler, local_PIPE.KSAMPLER.model_type)

        # TODO : make the feather_mask proportional to tile size ?
        # local_PIPE.PARAMS.feather_mask = local_PIPE.PARAMS.tile_size // 16
        local_PIPE.PARAMS.feather_mask = 0

        local_PIPE.OUTPUTS.output_info = ["No info"]
        local_PIPE.OUTPUTS.grid_tiles_to_process = []
        
        return local_PIPE
    
    @classmethod
    def _get_sigmas(cls, sigmas_type, model, steps, denoise, scheduler, model_type):
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
    # def upscale(cls, image):
        
    #     rows_qty_float = (image.shape[1] * cls.PARAMS.upscale_model_scale) / cls.PARAMS.tile_size
    #     cols_qty_float = (image.shape[2] * cls.PARAMS.upscale_model_scale) / cls.PARAMS.tile_size
    #     rows_qty = math.ceil(rows_qty_float)
    #     cols_qty = math.ceil(cols_qty_float)

    #     tiles_qty = rows_qty * cols_qty        
    #     if tiles_qty > Mara_Common_v1.MAX_TILES :
    #         msg = get_log(f"\n\n--------------------\n\n!!! Number of tiles is higher than {Mara_Common_v1.MAX_TILES} ({tiles_qty} for {cls.PARAMS.cols_qty} cols and {cls.PARAMS.rows_qty} rows)!!!\n\nPlease consider increasing your tile and feather sizes\n\n--------------------\n", "BLUE", "YELLOW", f"Node {cls.INFO.id} - Mara_McBoaty_Configurator_v6")
    #         raise ValueError(msg)

    #     upscaled_image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(cls.PARAMS.upscale_model, image)[0]

    #     cls.PARAMS.rows_qty = rows_qty
    #     cls.PARAMS.cols_qty = cols_qty
        
    #     # grid_specs = MS_Image().get_dynamic_grid_specs(upscaled_image.shape[2], upscaled_image.shape[1], rows_qty, cols_qty, cls.PARAMS.feather_mask)[0]
    #     grid_specs = MS_Image().get_tiled_grid_specs(upscaled_image, cls.PARAMS.tile_size, cls.PARAMS.rows_qty, cls.PARAMS.cols_qty, cls.PARAMS.feather_mask)[0]
    #     grid_images = MS_Image().get_grid_images(upscaled_image, grid_specs)
        
    #     grid_prompts = []
    #     llm = MS_Llm(cls.LLM.vision_model, cls.LLM.model)
    #     prompt_context = llm.vision_llm.generate_prompt(image)
    #     total = len(grid_images)
    #     for index, grid_image in enumerate(grid_images):
    #         prompt_tile = prompt_context
    #         if cls.PARAMS.tile_prompting_active:
    #             log(f"tile {index + 1}/{total} - [tile prompt]", None, None, f"Node {cls.INFO.id} - Prompting")
    #             prompt_tile = llm.generate_tile_prompt(grid_image, prompt_context, cls.KSAMPLER.noise_seed)
    #         log(f"tile {index + 1}/{total} - [tile prompt] {prompt_tile}", None, None, f"Node {cls.INFO.id} - Prompting")
    #         grid_prompts.append(prompt_tile)
                            
    #     return grid_specs, grid_images, grid_prompts

class Mara_McBoaty_Refiner_v6():
    
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
    def INPUT_TYPES(cls):
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
                "color_match_method": (cls.COLOR_MATCH_METHODS, { "label": "Color Match Method", "default": 'none'}),
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
        
        local_PIPE = self.init(**kwargs)
        local_PIPE.KSAMPLER.tiles = copy.deepcopy(local_PIPE.KSAMPLER.tiles)

        log("McBoaty (Refiner) is starting to do its magic", None, None, f"Node {local_PIPE.INFO.id}")

        tiles = kwargs.get('pipe_prompty', ([],))[0]
        if len(tiles) > 0:
            local_PIPE.KSAMPLER.tiles = Mara_Common_v1.override_tiles(local_PIPE, copy.deepcopy(local_PIPE.KSAMPLER.tiles), tiles)
        local_PIPE.KSAMPLER.tiles = self.refine(local_PIPE, local_PIPE.KSAMPLER.tiles)
        end_time = time.time()

        output_info = self._get_info(
            int(end_time - start_time)
        )

        mc_boaty_pipe = Mara_Common_v1.set_mc_boaty_pipe(local_PIPE)
        
        tiles = [t.new_tile for t in local_PIPE.KSAMPLER.tiles]
        cannies = [t.canny for t in local_PIPE.KSAMPLER.tiles]

        log("McBoaty (Refiner) is done with its magic", None, None, f"Node {local_PIPE.INFO.id}")

        return (
            mc_boaty_pipe,
            (
                copy.deepcopy(local_PIPE.KSAMPLER.tiles),
            ),            
            torch.cat(tiles, dim=0),
            torch.cat(cannies, dim=0),
            output_info, 
        )
        
    @classmethod
    def init(self, **kwargs):
        
        local_PIPE = Mara_Common_v1().init()

        pipe = kwargs.get('pipe', (SimpleNamespace(),) * len(Mara_Common_v1.PIPE_ATTRIBUTES))
        local_PIPE = Mara_Common_v1.set_pipe_values(local_PIPE, pipe)

        local_PIPE.INFO.id = kwargs.get('id', None)

        _tiles_to_process = kwargs.get('tiles_to_process', '')
        local_PIPE.PARAMS.tiles_to_process = Mara_McBoaty_Refiner_v6.set_tiles_to_process(local_PIPE, _tiles_to_process)
        local_PIPE.PARAMS.color_match_method = kwargs.get('color_match_method', 'none')
        
        return local_PIPE
                    
    @staticmethod
    def set_tiles_to_process(local_PIPE, tiles_to_process=''):

        max_tiles = len(local_PIPE.OUTPUTS.grid_tiles_to_process)
        max = max_tiles if max_tiles > 0 else Mara_Common_v1.MAX_TILES
        
        def is_valid_index(index, max = Mara_Common_v1.MAX_TILES):
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
                    log(f"tiles_to_process is not in valid format '{tiles_to_process}' - Allowed formats : indexes from 1 to {max} or any range like 1-{max}", None, COLORS['YELLOW'], f"Node {cls.INFO.id}")
            else:
                # Single index
                try:
                    index = int(index)
                    if is_valid_index(index, max):
                        _tiles_to_process.append(to_computer_index(index))
                    else:
                        _tiles_to_process.append(-1)
                        log(f"tiles_to_process is not in valid format '{tiles_to_process}' - Allowed formats : indexes from 1 to {max} or any range like 1-{max}", None, COLORS['YELLOW'], f"Node {cls.INFO.id}")
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
    def refine(self, local_PIPE, tiles):
        
        total = len(tiles)
        for index, tile in enumerate(tiles):
            _tile = tile.tile
            _tile = _tile.squeeze(0).permute(2, 0, 1)
            _, h, w = _tile.shape
            dim_max = max(h, w)
            if dim_max < local_PIPE.PARAMS.tile_size:
                dim_max = local_PIPE.PARAMS.tile_size
            else:
                _tile, _, _, local_PIPE.INFO.is_image_divisible_by_8 = MS_Image().format_2_divby8(image=tile.tile)
                _tile = _tile.squeeze(0).permute(2, 0, 1)
                _, h, w = _tile.shape
                dim_max = max(h, w)

            new_tile, tile_padding = MS_Image.pad_to_square(_tile, dim_max)
            new_tile = new_tile.permute(1, 2, 0).unsqueeze(0)
            if len(local_PIPE.PARAMS.tiles_to_process) == 0 or index in local_PIPE.PARAMS.tiles_to_process:
                if local_PIPE.KSAMPLER.tiled:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {local_PIPE.INFO.id} - VAEEncodingTiled")
                    tile.latent = nodes.VAEEncodeTiled().encode(local_PIPE.KSAMPLER.vae, new_tile, local_PIPE.KSAMPLER.tile_size_vae)[0]
                else:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {local_PIPE.INFO.id} - VAEEncoding")
                    tile.latent = nodes.VAEEncode().encode(local_PIPE.KSAMPLER.vae, new_tile)[0]
        
                sigmas = local_PIPE.KSAMPLER.sigmas
                if tile.denoise != local_PIPE.KSAMPLER.denoise:
                    denoise = tile.denoise
                    sigmas = Mara_McBoaty_Configurator_v6._get_sigmas(local_PIPE.KSAMPLER.sigmas_type, local_PIPE.KSAMPLER.model, local_PIPE.KSAMPLER.steps, tile.denoise, local_PIPE.KSAMPLER.scheduler, local_PIPE.KSAMPLER.model_type)
                else:
                    denoise = local_PIPE.KSAMPLER.denoise
                    
                log(f"tile {index + 1}/{total} : {denoise} / {tile.positive}", None, None, f"Node {local_PIPE.INFO.id} - Denoise/ClipTextEncoding")
                positive = nodes.CLIPTextEncode().encode(local_PIPE.KSAMPLER.clip, tile.positive)[0]
                negative = nodes.CLIPTextEncode().encode(local_PIPE.KSAMPLER.clip, tile.negative)[0]
                tile.controlnet = True
                if local_PIPE.CONTROLNET.controlnet is not None and tile.controlnet:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {local_PIPE.INFO.id} - ControlNetApply")
                    tile.canny = Canny().detect_edge(new_tile, local_PIPE.CONTROLNET.low_threshold, local_PIPE.CONTROLNET.high_threshold)[0]
                    positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive, negative, local_PIPE.CONTROLNET.controlnet, tile.canny, tile.strength, tile.start_percent, tile.end_percent, local_PIPE.KSAMPLER.vae )
                    
                log(f"tile {index + 1}/{total}", None, None, f"Node {local_PIPE.INFO.id} - Refining")
                _latent = tile.latent
                _latent = comfy_extras.nodes_custom_sampler.SamplerCustom().sample(
                    local_PIPE.KSAMPLER.model, 
                    local_PIPE.KSAMPLER.add_noise, 
                    local_PIPE.KSAMPLER.noise_seed, 
                    tile.cfg, 
                    positive,
                    negative,
                    local_PIPE.KSAMPLER.sampler, 
                    sigmas, 
                    _latent
                )[0]
                tile.latent = _latent

                if local_PIPE.KSAMPLER.tiled:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {local_PIPE.INFO.id} - VAEDecodingTiled")
                    new_tile = nodes.VAEDecodeTiled().decode(local_PIPE.KSAMPLER.vae, tile.latent, local_PIPE.KSAMPLER.tile_size_vae, int(local_PIPE.KSAMPLER.tile_size_vae * local_PIPE.PARAMS.overlap))[0]
                else:
                    log(f"tile {index + 1}/{total}", None, None, f"Node {local_PIPE.INFO.id} - VAEDecoding")
                    new_tile = nodes.VAEDecode().decode(local_PIPE.KSAMPLER.vae, tile.latent)[0]
                _new_tile = new_tile
                new_tile = new_tile.squeeze(0).permute(2, 0, 1)
                new_tile = MS_Image.crop_to_original(new_tile, _tile.shape, tile_padding)
                tile.new_tile = new_tile.permute(1, 2, 0).unsqueeze(0)
        
        return tiles

class Mara_McBoaty_TilePrompter_v6():

    NAME = "McBoaty Tile Prompter"
    SHORTCUT = "m"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "pipe_prompty": ("MC_PROMPTY_PIPE", {"label": "McPrompty Pipe" }),
                "tiles_to_process": ("STRING", { "label": "Tile to process", "default": ""}),
                "positive": ("STRING", { "label": "Positive (Prompt)", "multiline": True, "default": Mara_Common_v1.TILE_ATTRIBUTES.positive }),
                "negative": ("STRING", { "label": "Negative (Prompt)", "multiline": True, "default": Mara_Common_v1.TILE_ATTRIBUTES.negative }),
                "cfg": ("FLOAT", { "label": "CFG", "default": Mara_Common_v1.TILE_ATTRIBUTES.cfg, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "denoise": ("FLOAT", { "label": "Denoise", "default": Mara_Common_v1.TILE_ATTRIBUTES.denoise, "min": 0.0, "max": 1.0, "step": 0.01}),
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
    def fn(cls, **kwargs):
        
        start_time = time.time()

        tiles = kwargs.get('pipe_prompty', ([],))[0]
        tiles = copy.deepcopy(tiles)
        
        nodeid = kwargs.get('id', None)
        
        log("McBoaty (PromptEditor) is starting to do its magic", None, None, f"Node {nodeid}")
        
        tile_attributes = copy.deepcopy(Mara_Common_v1.TILE_ATTRIBUTES)

        attributes = {
            'positive': kwargs.get('positive', tile_attributes.positive),
            'negative': kwargs.get('negative', tile_attributes.negative),
            'cfg': round(kwargs.get('cfg', tile_attributes.cfg), 2),
            'denoise': round(kwargs.get('denoise', tile_attributes.denoise), 2),
            'strength': round(kwargs.get('strength', tile_attributes.strength), 2),
            'start_percent': round(kwargs.get('start_percent', tile_attributes.start_percent), 3),
            'end_percent': round(kwargs.get('end_percent', tile_attributes.end_percent), 3)
        }
        tiles_to_process = Mara_Common_v1.parse_tiles_to_process(kwargs.get('tiles_to_process', ""), len(tiles))
        
        if not tiles_to_process:  # This works for empty lists/arrays
            tiles_to_process = list(range(1, len(tiles) + 1))

        for id in tiles_to_process:
            index = id - 1
            for attr, value in attributes.items():
                if value != getattr(tile_attributes, attr) and value != getattr(tiles[index], attr) and value != '':
                    setattr(tiles[index], attr, value)
            store.add_item(nodeid, index, {"positive": tiles[index].positive, "negative": tiles[index].negative})

        end_time = time.time()
        preocess_time = int(end_time - start_time)

        log("McBoaty (PromptEditor) is done with its magic", None, None, f"Node {id}")
        
        return (
            (
                tiles,
            ),
        )

                
class Mara_McBoaty_v6():

    NAME = "McBoaty"
    SHORTCUT = "m"

    @classmethod
    def INPUT_TYPES(cls):
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
    def fn(cls, **kwargs):
        
        start_time = time.time()
        
        local_PIPE = cls.init(**kwargs)
        
        local_PIPE.INFO.id = kwargs.get('id', None)
        
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
        combined_info = cls._combine_info(upscaler_info, refiner_info, total_time)
        
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

        local_PIPE = Mara_Common_v1().init()

        local_PIPE = Mara_McBoaty_Configurator_v6.init(**kwargs)
        
        _tiles_to_process = kwargs.get('tiles_to_process', '')
        local_PIPE.PARAMS.tiles_to_process = Mara_McBoaty_Refiner_v6.set_tiles_to_process(local_PIPE, _tiles_to_process)
        local_PIPE.PARAMS.color_match_method = kwargs.get('color_match_method', 'none')
        
        return local_PIPE