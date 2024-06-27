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
import torch
import math
from types import SimpleNamespace
import comfy
import comfy_extras
import comfy_extras.nodes_custom_sampler
from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
import nodes
from PIL import Image
from server import PromptServer
from aiohttp import web
import folder_paths

from .... import root_dir
from ...utils.version import VERSION
from ...inc.lib.image import MS_Image_v2 as MS_Image
from ...vendor.ComfyUI_KJNodes.nodes.image_nodes import ColorMatch as ColorMatch
from ...inc.lib.llm import MS_Llm
from ...inc.lib.cache import MS_Cache

from .inc.prompt import Node as NodePrompt

from ...utils.log import log, COLORS


class McBoaty_Upscaler_v4():

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
        "STRING",
        "IMAGE",
        "STRING",
    )
    
    RETURN_NAMES = (
        "pipe",
        "prompts", 
        "tiles (Upscaled)",
        "info", 
    )
    
    OUTPUT_IS_LIST = (
        False,
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
            raise ValueError("MaraScottUpscalerRefinerNode id XX: No image provided")

        if not isinstance(self.INPUTS.image, torch.Tensor):
            raise ValueError("MaraScottUpscalerRefinerNode id XX: Image provided is not a Tensor")
        
        log("McBoaty (Upscaler) is starting to do its magic")
        
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
        
        log("McBoaty (Upscaler) is done with its magic")

        output_tiles = torch.cat(self.OUTPUTS.grid_images)

        return (
            (
                self.INPUTS,
                self.PARAMS,
                self.KSAMPLER,
                self.OUTPUTS,
            ),
            self.OUTPUTS.grid_prompts,
            output_tiles,
            output_info
        )
        
    @classmethod
    def init(self, **kwargs):
        # Initialize the bus tuple with None values for each parameter
        self.INPUTS = SimpleNamespace(
            image = kwargs.get('image', None),
        )
        
        self.LLM = SimpleNamespace(
            vision_model = kwargs.get('vision_llm_model', None),
            model = kwargs.get('llm_model', None),
        )
        
        self.PARAMS = SimpleNamespace(
            upscale_size_type = kwargs.get('output_size_type', None),
            upscale_size = kwargs.get('output_size', None),
            upscale_model_name = kwargs.get('upscale_model', None),
            upscale_method = kwargs.get('output_upscale_method', "lanczos"),
            feather_mask = kwargs.get('feather_mask', None),
            color_match_method = kwargs.get('color_match_method', 'none'),
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
        self.KSAMPLER.tile_size_sampler = self.AYS_MODEL_TYPE_SIZES[self.KSAMPLER.ays_model_type]
        self.KSAMPLER.sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, self.KSAMPLER.denoise, self.KSAMPLER.scheduler, self.KSAMPLER.ays_model_type)
        self.KSAMPLER.outpaint_sigmas = self._get_sigmas(self.KSAMPLER.sigmas_type, self.KSAMPLER.model, self.KSAMPLER.steps, 1, self.KSAMPLER.scheduler, self.KSAMPLER.ays_model_type)

        # TODO : make the feather_mask proportional to tile size ?
        # self.PARAMS.feather_mask = self.KSAMPLER.tile_size // 16

        self.OUTPUTS = SimpleNamespace(
            grid_images = [],
            grid_prompts = [],
            output_info = ["No info"],
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
    def upscale(self, image, iteration):
        
        feather_mask = self.PARAMS.feather_mask

        upscaled_image = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(self.PARAMS.upscale_model, image)[0]

        rows_qty_float = upscaled_image.shape[1] / self.KSAMPLER.tile_size
        cols_qty_float = upscaled_image.shape[2] / self.KSAMPLER.tile_size
        self.PARAMS.rows_qty = math.ceil(rows_qty_float)
        self.PARAMS.cols_qty = math.ceil(cols_qty_float)
        
        # grid_specs = MS_Image().get_dynamic_grid_specs(upscaled_image.shape[2], upscaled_image.shape[1], rows_qty, cols_qty, feather_mask)[0]
        grid_specs = MS_Image().get_tiled_grid_specs(upscaled_image, self.KSAMPLER.tile_size, self.PARAMS.rows_qty, self.PARAMS.cols_qty, feather_mask)[0]
        grid_images = MS_Image().get_grid_images(upscaled_image, grid_specs)
        
        grid_prompts = []
        llm = MS_Llm(self.LLM.vision_model, self.LLM.model)
        prompt_context = llm.vision_llm.generate_prompt(image)
        total = len(grid_images)
        for index, grid_image in enumerate(grid_images):
            prompt_tile = "beautiful"
            if self.PARAMS.tile_prompting_active:
                log(f"tile {index + 1}/{total} - [tile prompt]", None, None, f"Prompting {iteration}")
                prompt_tile = llm.generate_tile_prompt(grid_image, prompt_context, self.KSAMPLER.noise_seed)
                log(f"tile {index + 1}/{total} - [tile prompt] {prompt_tile}", None, None, f"Prompting {iteration}")

            grid_prompts.append(prompt_tile)
                            
        return grid_specs, grid_images, grid_prompts

class McBoaty_Refiner_v4():

    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "pipe": ("MC_BOATY_PIPE", {"label": "Mc Boaty Pipe" }),
            },
            "optional": {
                "prompts": ("STRING", {"label": "Prompts", "forceInput": True }),
                "tiles": ("IMAGE", {"label": "Tiles" }),
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
    
    OUTPUT_IS_LIST = (
        False,
        False,
        False,
        False,
    )
    
    
    OUTPUT_NODE = True
    CATEGORY = "MaraScott/upscaling"
    DESCRIPTION = "A \"Refiner\" Node"
    FUNCTION = "fn"

    @classmethod    
    def fn(self, **kwargs):
        
        start_time = time.time()
        
        self.init(**kwargs)

        log("McBoaty (Refiner) is starting to do its magic")
        
        self.PARAMS.grid_prompts, self.OUTPUTS.output_image, self.OUTPUTS.output_tiles = self.refine(self.OUTPUTS.image, "Upscaling")
            
        end_time = time.time()

        output_info = self._get_info(
            int(end_time - start_time)
        )
        
        log("McBoaty (Refiner) is done with its magic")
        
        return (
            self.OUTPUTS.output_image, 
            self.OUTPUTS.output_tiles, 
            self.OUTPUTS.image, 
            output_info, 
        )
        
    @classmethod
    def init(self, **kwargs):
        attribute_names = ('INPUTS', 'PARAMS', 'KSAMPLER', 'OUTPUTS') 
        pipe = kwargs.get('pipe', (None,) * len(attribute_names))

        for name, value in zip(attribute_names, pipe):
            setattr(self, name, value)
            
        self.PARAMS.tile_prompting_active = True
        
        grid_images = kwargs.get('tiles', (None,) * len(self.OUTPUTS.grid_images))
        grid_images = list(grid_images)
        for i, image in enumerate(grid_images):
            if image is None:
                grid_images[i] = self.OUTPUTS.grid_images[i]
        self.OUTPUTS.grid_images = grid_images

        grid_prompts = kwargs.get('prompts', (None,) * len(self.OUTPUTS.grid_prompts))
        if len(grid_prompts) != len(self.OUTPUTS.grid_prompts):
            grid_prompts = [gp if gp is not None else default_gp for gp, default_gp in zip(grid_prompts, self.OUTPUTS.grid_prompts)]
        grid_prompts = list(grid_prompts)
        for i, prompt in enumerate(grid_prompts):
            if prompt is None:
                grid_prompts[i] = self.OUTPUTS.grid_prompts[i]
        self.OUTPUTS.grid_prompts = grid_prompts
            
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
            if self.KSAMPLER.tiled:
                log(f"tile {index + 1}/{total}", None, None, f"VAEEncodingTiled {iteration}")
                latent_image = nodes.VAEEncodeTiled().encode(self.KSAMPLER.vae, upscaled_image_grid, self.KSAMPLER.tile_size_vae)[0]
            else:
                log(f"tile {index + 1}/{total}", None, None, f"VAEEncoding {iteration}")
                latent_image = nodes.VAEEncode().encode(self.KSAMPLER.vae, upscaled_image_grid)[0]
            grid_latents.append(latent_image)
        
        for index, latent_image in enumerate(grid_latents):
            positive = self.KSAMPLER.positive
            if self.PARAMS.tile_prompting_active:
                log(f"tile {index + 1}/{total} : {self.OUTPUTS.grid_prompts[index]}", None, None, f"ClipTextEncoding {iteration}")
                positive = nodes.CLIPTextEncode().encode(self.KSAMPLER.clip, self.OUTPUTS.grid_prompts[index])[0]
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
            if self.KSAMPLER.tiled:
                log(f"tile {index + 1}/{total}", None, None, f"VAEDecodingTiled {iteration}")
                output = (nodes.VAEDecodeTiled().decode(self.KSAMPLER.vae, latent_output, self.KSAMPLER.tile_size_vae)[0].unsqueeze(0))[0]
            else:
                log(f"tile {index + 1}/{total}", None, None, f"VAEDecoding {iteration}")
                output = (nodes.VAEDecode().decode(self.KSAMPLER.vae, latent_output)[0].unsqueeze(0))[0]
            
            # output = nodes.ImageScaleBy().upscale(output, self.PARAMS.upscale_method, (1/(output.shape[2] / self.KSAMPLER.tile_size_sampler)))[0]
            output_images.append(output)

        feather_mask = self.PARAMS.feather_mask
        output_image, tiles_order = MS_Image().rebuild_image_from_parts(iteration, output_images, image, self.PARAMS.grid_specs, feather_mask, self.PARAMS.upscale_model.scale, self.PARAMS.rows_qty, self.PARAMS.cols_qty, self.OUTPUTS.grid_prompts)

        if self.PARAMS.color_match_method != 'none':
            output_image = ColorMatch().colormatch(image, output_image, self.PARAMS.color_match_method)[0]

        tiles_order.sort(key=lambda x: x[0])
        output_tiles = tuple(output for _, output, _ in tiles_order)
        output_tiles = torch.cat(output_tiles)
        output_prompts = tuple(prompt for _, _, prompt in tiles_order)

        return output_prompts, output_image, output_tiles

class McBoaty_TilePrompter_v4():

    @classmethod
    def INPUT_TYPES(self):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
            },
            "required":{
                "prompts": ("STRING", {"label": "prompts" , "forceInput": True }),
            },
            "optional": {
                **NodePrompt.ENTRIES,
            }
        }

    RETURN_TYPES = (
        "STRING",
    )
    
    RETURN_NAMES = (
        "prompts",
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
        
        self.id = kwargs.get('id', 0)
        input_prompts = kwargs.get('prompts', ["No prompt"])

        self.init(self.id)
        
        _input_prompts = MS_Cache.get(self.cache_name, input_prompts)
        _input_prompts_edited = MS_Cache.get(self.cache_name_edited, input_prompts)
        
        if not MS_Cache.isset(self.cache_name) or _input_prompts != input_prompts:
            _input_prompts = input_prompts
            MS_Cache.set(self.cache_name, _input_prompts)

        if not MS_Cache.isset(self.cache_name_edited):
            MS_Cache.set(self.cache_name_edited, _input_prompts)
        elif len(_input_prompts_edited) != len(_input_prompts):
            _input_prompts_edited = [gp if gp is not None else default_gp for gp, default_gp in zip(_input_prompts_edited, _input_prompts)]
            MS_Cache.set(self.cache_name_edited, _input_prompts_edited)

        if _input_prompts_edited != _input_prompts:                
            input_prompts = _input_prompts_edited

        output_prompts = input_prompts
                
        return {"ui": {"prompts": output_prompts}, "result": (output_prompts,)}

    @classmethod
    def init(self, id = 0):
        self.cache_name = f'input_prompts_{id}'
        self.cache_name_edited = f'{self.cache_name}_edited'
        
    
@PromptServer.instance.routes.get("/MaraScott/McBoaty/v4/set_prompt")
async def set_prompt(request):
    prompt = request.query.get("prompt", None)
    index = int(request.query.get("index", -1))
    nodeId = request.query.get("node", None)
    clientId = request.query.get("clientId", None)
    cache_name = f'input_prompts_{nodeId}'
    cache_name_edited = f'{cache_name}_edited'
    _input_prompts = MS_Cache.get(cache_name, [])
    _input_prompts_edited = MS_Cache.get(cache_name_edited, _input_prompts)
    if _input_prompts_edited and index < len(_input_prompts_edited):
        _input_prompts_edited[index] = prompt
        MS_Cache.set(cache_name_edited, _input_prompts_edited)
    return web.json_response(f"Tile {index} prompt has been updated\n{prompt}")

@PromptServer.instance.routes.get("/MaraScott/McBoaty/v4/tile_prompt")
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

    # image = Image.open(image_path)

    return web.json_response(f"here is the prompt \n{image_path}")
