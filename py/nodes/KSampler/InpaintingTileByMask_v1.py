#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#       Inpaint by Mask
#
#       Largely inspired by PYSSSSS - ShowText 
#
###

import torch
from types import SimpleNamespace
import comfy
from comfy_extras import nodes_differential_diffusion as DiffDiff, nodes_images as extra_images, nodes_mask as extra_mask, nodes_compositing as extra_compo, nodes_upscale_model as extra_upscale_model
from nodes import KSampler, CLIPTextEncode, VAEEncodeTiled, VAEDecodeTiled, ImageScale, SetLatentNoiseMask, ImageScaleBy
import folder_paths

from ...inc.lib.image import MS_Image
from ...inc.lib.mask import MS_Mask

from ...vendor.ComfyUI_LayerStyle.py.image_blend_v2 import ImageBlendV2, chop_mode_v2
from ...vendor.ComfyUI_LayerStyle.py.image_opacity import ImageOpacity
from ...vendor.was_node_suite_comfyui.WAS_Node_Suite import WAS_Mask_Crop_Region, WAS_Image_Blend
from ...vendor.ComfyUI_Impact_Pack.modules.impact.util_nodes import RemoveNoiseMask
from ...vendor.mikey_nodes.mikey_nodes import ImagePaste
from ...vendor.ComfyUI_tinyterraNodes.ttNpy.tinyterraNodes import ttN_imageREMBG

from ...utils.log import *

class KSampler_setInpaintingTileByMask_v1:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", { "label": "Image" }),
                "painted_image": ("IMAGE", { "label": "Image (Painted)" }),
                "mask": ("MASK", { "label": "Mask (Painted)" }),
                "noise_image": ("IMAGE", { "label": "Image (Noise)" }),

                "inpaint_size": ("INT", { "label": "Inpaint Size", "default": 1024, "min": 512, "max": 1024, "step": 512}),
                "noise_blend": ("INT", { "label": "Blend (Noise)", "default": 100, "min": 0, "max": 100, "step": 1 }),
                "tile_blend_mode": (chop_mode_v2, { "label": "Blend Mode (tile)", "default": "normal" }),  # normal|overlay|pin light|dissolve|hue|linear light
                "tile_opacity": ("INT", { "label": "Opacity (tile)", "default": 100, "min": 0, "max": 100, "step": 1 }),

                "model": ("MODEL", { "label": "Model" }),
                "model_diffdiff": ("BOOLEAN", {"label": "Apply DiffDiff", "default": True}),
                "clip": ("CLIP", { "label": "Clip" }),
                "vae": ("VAE", { "label": "VAE" }),

                "text_pos_image": ("STRING", { "label": "Positive (text img)", "forceInput": True }),
                "text_pos_inpaint": ("STRING", { "label": "Positive (text painted)", "forceInput": True }),
                "text_neg_inpaint": ("STRING", { "label": "Negative (text img)", "forceInput": True }),

                "seed": ("INT", { "label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", { "label": "Steps", "default": 10, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", { "label": "CFG", "default": 8, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, { "label": "Sampler Name" }),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, { "label": "Basic Scheduler" }),
                "denoise": ("FLOAT", { "label": "Denoise", "default": 0.51, "min": 0.0, "max": 1.0, "step": 0.01}),

            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    # INPUT_IS_LIST = False
    RETURN_TYPES = (
        'IMAGE',
        'IMAGE',
        'STRING',
        'STRING',
        'MS_INPAINTINGTILEBYMASK_PIPE',
    )
    RETURN_NAMES = (
        'image_inpainted',
        'image_noised',
        'text_pos_image_inpainted',
        'text_neg_image_inpainted',
        'ms_pipe',
    )
    RETURN_LABELS = (
        'Inpainted Tile',
        'Noised Tile',
        'positive (text)',
        'negative (text)',
        'pipe (InpaintingTileByMask)',
    )
    OUTPUT_IS_LIST = (
        False,
        False,
        False,
        False,
        False,
    )
    FUNCTION = "fn"
    OUTPUT_NODE = True

    CATEGORY = "MaraScott/Ksampler"

    def fn(s, **kwargs):
        s.init(**kwargs)
        s.format_inputs()

        if torch.all(s.inputs.painted_mask == 0):
            s.set_default_outputs()
        else:
            s.set_tile_region()
            s.crop_tiles()
            s.upscale_tiles()
            s.set_painted_image_cropped_noised()
            s.set_tile_noise_by_mask()
            s.ksample_tile()

            s.text_pos_image_inpainted = f"{s.inputs.text_pos_image}, {s.inputs.text_pos_inpaint}"
            s.text_neg_image_inpainted = s.inputs.text_neg_inpaint

            pipe = (
                s.inputs.source,
                s.inputs.painted,
                s.inputs.painted_mask,
                s.inputs.noise,
                s.ksampler.model, 
                s.params.is_model_diffdiff, 
                s.ksampler.clip, 
                s.ksampler.vae, 
                s.text_pos_image_inpainted, 
                s.text_neg_image_inpainted,
                s.params.mask_region.mask_cropped,
                s.tile.source,
                s.params.mask_region.width,
                s.params.mask_region.height,
                s.params.mask_region.x,
                s.params.mask_region.y,
                s.params.inpaint_size,
            )
        
        return (
            s.tile.inpainted,
            s.tile.noised_by_mask,
            s.text_pos_image_inpainted, 
            s.text_neg_image_inpainted,
            pipe,
        )

    def init(s, **kwargs):
        s.inputs = SimpleNamespace(
            source = kwargs.get('image', None),
            painted = kwargs.get('painted_image', None),
            painted_mask = kwargs.get('mask', None),
            noise = kwargs.get('noise_image', None),
            text_pos_image = kwargs.get('text_pos_image', None),
            text_pos_inpaint = kwargs.get('text_pos_inpaint', None),
            text_neg_inpaint = kwargs.get('text_neg_inpaint', None),
        )
        s.params = SimpleNamespace(
            is_model_diffdiff = kwargs.get('model_diffdiff', True),
            upscale_method = "lanczos",    
            inpaint_size = kwargs.get('inpaint_size', None),
            noise_blend = kwargs.get('noise_blend', None),
            tile_blend_mode = kwargs.get('tile_blend_mode', None),
            tile_opacity = kwargs.get('tile_opacity', None),
        )
        s.ksampler = SimpleNamespace(
            model = kwargs.get('model', None),
            clip = kwargs.get('clip', None),
            vae = kwargs.get('vae', None),
            seed = kwargs.get('seed', None),
            steps = kwargs.get('steps', None),
            cfg = kwargs.get('cfg', None),
            sampler_name = kwargs.get('sampler_name', None),
            scheduler = kwargs.get('basic_scheduler', None),
            denoise = kwargs.get('denoise', None),
        )
        s.tile = SimpleNamespace()

        s.ksampler.model_inpaint = DiffDiff.DifferentialDiffusion().apply(s.ksampler.model)[0] if s.params.is_model_diffdiff else s.ksampler.model
        s.ksampler.positive_inpaint = CLIPTextEncode().encode(s.ksampler.clip, s.inputs.text_pos_inpaint)[0]
        s.ksampler.negative_inpaint = CLIPTextEncode().encode(s.ksampler.clip, s.inputs.text_neg_inpaint)[0]


    def format_inputs(s):

        # format and get size from input image
        s.inputs.source, s.image_width, s.image_height, s.image_divisible_by_8 = MS_Image().format_2_divby8(s.inputs.source)

        # Upscale and process painted image and mask
        s.inputs.painted = ImageScale().upscale(s.inputs.painted, s.params.upscale_method, s.image_width, s.image_height, "center")[0]

        mask_image = extra_mask.MaskToImage().mask_to_image(s.inputs.painted_mask)[0]
        mask_image = ImageScale().upscale(mask_image, s.params.upscale_method, s.image_width, s.image_height, "center")[0]
        s.inputs.painted_mask = extra_mask.ImageToMask().image_to_mask(mask_image, 'red')[0]

        s.inputs.noise = ImageScale().upscale(s.inputs.noise, s.params.upscale_method, s.image_width, s.image_height, "center")[0]

    def set_default_outputs(s):
        s.tile.noised_by_mask = s.inputs.source
        s.tile.noised_by_mask = s.inputs.noise
        s.text_pos_image_inpainted = s.inputs.text_pos_image
        s.text_neg_image_inpainted = s.inputs.text_neg_inpaint
        
        s.pipe = (
            s.inputs.source,
            s.inputs.painted,
            s.inputs.painted_mask,
            s.inputs.noise,
            s.ksampler.model, 
            s.params.is_model_diffdiff, 
            s.ksampler.clip, 
            s.ksampler.vae, 
            s.text_pos_image_inpainted, 
            s.text_neg_image_inpainted,
            MS_Mask.empty(s.image_width, s.image_height),
            MS_Image.empty(s.image_width, s.image_height),
            s.image_width,
            s.image_height,
            0, 
            0,
            s.params.inpaint_size
        )


    def set_tile_region(s):

        region = WAS_Mask_Crop_Region().mask_crop_region(s.inputs.painted_mask, padding=0, region_type="dominant")
        s.params.mask_region = SimpleNamespace(
            mask_cropped = region[0],
            x = region[3],
            y = region[2],
            width = region[6],
            height = region[7],
        )

    def crop_tiles(s):

        # image
        s.tile.source = extra_images.ImageCrop().crop(s.inputs.source, s.params.mask_region.width, s.params.mask_region.height, s.params.mask_region.x, s.params.mask_region.y)[0]
        s.tile.painted = extra_images.ImageCrop().crop(s.inputs.painted, s.params.mask_region.width, s.params.mask_region.height, s.params.mask_region.x, s.params.mask_region.y)[0]
        s.tile.painted_mask = s.params.mask_region.mask_cropped
        s.tile.noise = extra_images.ImageCrop().crop(s.inputs.noise, s.params.mask_region.width, s.params.mask_region.height, s.params.mask_region.x, s.params.mask_region.y)[0]

    def set_painted_image_cropped_noised(s):
        # Blend painted image with noise image
        s.tile.noised = WAS_Image_Blend().image_blend(
            image_a=s.tile.painted, 
            image_b=s.tile.noise, 
            blend_percentage=s.params.noise_blend
        )[0]

    def upscale_tiles(s):

        # Image Upscale
        s.tile.source = ImageScale().upscale(s.tile.source, s.params.upscale_method, s.params.inpaint_size, s.params.inpaint_size, "disabled")[0]

        # Mask Upscale
        painted_mask = extra_mask.MaskToImage().mask_to_image(s.tile.painted_mask)[0]
        painted_mask = ImageScale().upscale(painted_mask, s.params.upscale_method, s.params.inpaint_size, s.params.inpaint_size, "center")[0]
        s.tile.painted_mask = extra_mask.ImageToMask().image_to_mask(painted_mask, 'red')[0]

        # Noise Upscale
        s.tile.noise = ImageScale().upscale(s.tile.noise, s.params.upscale_method, s.params.inpaint_size, s.params.inpaint_size, "center")[0]

    def set_tile_noise_by_mask(s):
        s.tile.noised_by_mask = ImageBlendV2().image_blend_v2(
            background_image=s.tile.source, 
            layer_image=s.tile.noise, 
            invert_mask=False, 
            blend_mode=s.params.tile_blend_mode, 
            opacity=s.params.tile_opacity, 
            layer_mask=s.tile.painted_mask
        )[0]

    def ksample_tile(s):
        
        latent = VAEEncodeTiled().encode(s.ksampler.vae, s.tile.noised_by_mask, tile_size=512)[0]
        if s.params.is_model_diffdiff:
            latent = SetLatentNoiseMask().set_mask(latent, s.params.mask_region.mask_cropped)[0]

        latent = KSampler().sample(
            model=s.ksampler.model,
            seed=s.ksampler.seed, 
            steps=s.ksampler.steps, 
            cfg=s.ksampler.cfg, 
            sampler_name=s.ksampler.sampler_name, 
            scheduler=s.ksampler.scheduler, 
            positive=s.ksampler.positive_inpaint, 
            negative=s.ksampler.negative_inpaint, 
            latent_image=latent, 
            denoise=s.ksampler.denoise
        )[0]
        
        if s.params.is_model_diffdiff:
            latent = RemoveNoiseMask().doit(latent)[0]
        s.tile.inpainted = VAEDecodeTiled().decode(s.ksampler.vae, latent, tile_size=512)[0]

class KSampler_pasteInpaintingTileByMask_v1:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_tile": ("IMAGE", { "label": "Image (Tile)" }),
                "mask_tile": ("MASK", { "label": "Mask (Tile)" }),
                "upscale_by_model": ("BOOLEAN", {"label": "Upscale By Model", "default": False}),
                "upscale_model": (folder_paths.get_filename_list("upscale_models"),),
                "ms_pipe": ("MS_INPAINTINGTILEBYMASK_PIPE", { "label": "pipe (InpaintingTileByMask)" }),
                "text_pos_inpaint": ("STRING", { "label": "Positive (text) optional" }),
                "text_neg_inpaint": ("STRING", { "label": "Negative (text) optional" }),

                "subject_opacity": ("INT", { "label": "Opacity (Mask)", "default": 95, "min": 0, "max": 100, "step": 1 }),

                "seed": ("INT", { "label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", { "label": "Steps", "default": 10, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", { "label": "CFG", "default": 8, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, { "label": "Sampler Name" }),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, { "label": "Basic Scheduler" }),
                "denoise": ("FLOAT", { "label": "Denoise", "default": 0.51, "min": 0.0, "max": 1.0, "step": 0.01}),

            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        'IMAGE',
        'IMAGE',
        'IMAGE',
        'STRING',
        'STRING',
    )
    RETURN_NAMES = (
        'output_image',
        'image_inpainted',
        'image',
        'text_pos_image_inpainted',
        'text_neg_image_inpainted',
    )
    RETURN_LABELS = (
        'Image',
        'Image Inpainting (input)',
        'Image (Original)',
        'positive (text)',
        'negative (text)',
    )
    FUNCTION = "fn"
    OUTPUT_NODE = True

    CATEGORY = "MaraScott/Ksampler"

    def fn(s, **kwargs):
        
        s.init(**kwargs)
        s.inputs.source, s.image_width, s.image_height, s.image_divisible_by_8 = MS_Image().format_2_divby8(s.inputs.source)

        if torch.all(s.inputs.painted_mask == 0):
            
            output_image = s.inputs.source
            image_inpainted = s.inputs.painted
            image = s.inputs.source
            text_pos_image_inpainted = s.inputs.text_pos_inpaint
            text_neg_image_inpainted = s.inputs.text_neg_inpaint
        
        else:

            s.set_subject_region()
            s.paste_subject2tile()
            # s.outputs.image = ImageScale().upscale(s.tile.inpainted, s.params.upscale_method, s.params.mask_region.width, s.params.mask_region.height, "center")[0]
            
            s.refine_tile()
            s.paste_tile2source()
            
            # latent = VAEEncodeTiled().encode(s.ksampler.vae, s.outputs.image, tile_size=512)[0]
            # if s.params.is_model_diffdiff:
            #     latent = SetLatentNoiseMask().set_mask(latent, s.inputs.painted_mask)[0]            

            # latent = KSampler().sample(
            #     model=s.ksampler.model_inpaint, 
            #     seed=s.ksampler.seed, 
            #     steps=s.ksampler.steps, 
            #     cfg=s.ksampler.cfg, 
            #     sampler_name=s.ksampler.sampler_name, 
            #     scheduler=s.ksampler.scheduler, 
            #     positive=s.ksampler.positive_inpaint, 
            #     negative=s.ksampler.negative_inpaint, 
            #     latent_image=latent, 
            #     denoise=s.ksampler.denoise
            # )[0]

            # if s.params.is_model_diffdiff:
            #     latent = RemoveNoiseMask().doit(latent)[0]
            # s.outputs.image = VAEDecodeTiled().decode(s.ksampler.vae, latent, tile_size=512)[0]

            s.outputs.image = ImageScale().upscale(s.outputs.image, s.params.upscale_method, s.image_width, s.image_height, "center")[0]

            output_image = s.outputs.image
            image_inpainted = s.inputs.painted
            image = s.inputs.source
            text_pos_image_inpainted = s.inputs.text_pos_inpaint
            text_neg_image_inpainted = s.inputs.text_neg_inpaint
            
        return (
            output_image,
            image_inpainted,
            image,
            text_pos_image_inpainted, 
            text_neg_image_inpainted
        )
        
    def init(s, **kwargs):
        
        s.unique_id = kwargs.get('unique_id', None)

        s.tile = SimpleNamespace(
            image = kwargs.get('image_tile', None),
            mask = kwargs.get('mask_tile', None),
        )

        s.inputs = SimpleNamespace(
            tile = s.tile
        )
        s.outputs = SimpleNamespace(
        )

        s.params = SimpleNamespace(
            upscale_method = "lanczos",    
            upscale_model_name = kwargs.get('upscale_model', None),
            upscale_by_model = kwargs.get('upscale_by_model', None),
            subject_opacity = kwargs.get('subject_opacity', None),
            mask_region = SimpleNamespace(),
        )
                
        s.ksampler = SimpleNamespace(
            seed = kwargs.get('seed', None),
            steps = kwargs.get('steps', None),
            cfg = kwargs.get('cfg', None),
            sampler_name = kwargs.get('sampler_name', None),
            scheduler = kwargs.get('basic_scheduler', None),
            denoise = kwargs.get('denoise', None),                        
        )

        ms_pipe = kwargs.get('ms_pipe', None)
        s.inputs.source, s.inputs.painted, s.inputs.painted_mask, s.inputs.noise, s.ksampler.model, s.params.is_model_diffdiff, s.ksampler.clip, s.ksampler.vae, s.inputs.text_pos_inpaint, s.inputs.text_neg_inpaint, s.params.mask_region.mask_cropped, s.tile.source, s.params.mask_region.width, s.params.mask_region.height, s.params.mask_region.x, s.params.mask_region.y, s.params.inpaint_size = ms_pipe

        s.inputs.text_pos_inpaint = kwargs.get('text_pos_inpaint', s.inputs.text_pos_inpaint)
        s.inputs.text_neg_inpaint = kwargs.get('text_neg_inpaint', s.inputs.text_neg_inpaint)
        s.ksampler.model_inpaint = DiffDiff.DifferentialDiffusion().apply(s.ksampler.model)[0] if s.params.is_model_diffdiff else s.ksampler.model
        s.ksampler.positive_inpaint = CLIPTextEncode().encode(s.ksampler.clip, s.inputs.text_pos_inpaint)[0]
        s.ksampler.negative_inpaint = CLIPTextEncode().encode(s.ksampler.clip, s.inputs.text_neg_inpaint)[0]

    def set_subject_region(s):

        region = WAS_Mask_Crop_Region().mask_crop_region(s.inputs.tile.mask, padding=0, region_type="dominant")
        
        s.params.subject_region = SimpleNamespace(
            mask_cropped = region[0],
            x = region[3],
            y = region[2],
            width = region[6],
            height = region[7],
        )

    def paste_subject2tile(s):
        subject_tile = extra_mask.ImageCompositeMasked().composite(s.inputs.tile.source, s.inputs.tile.source, x = s.params.subject_region.x, y = s.params.subject_region.y, resize_source = False, mask = s.inputs.tile.mask)[0]
        subject_tile = ImageOpacity().image_opacity(image=subject_tile, opacity=100, invert_mask=False)[0]
        s.tile.inpainted = ImageScale().upscale(subject_tile, s.params.upscale_method, s.params.mask_region.width, s.params.mask_region.height, "center")[0]

    def refine_tile(s):
        
        inpainted = ImageScaleBy().upscale(s.tile.inpainted, s.params.upscale_method, 1.5)[0]
        mask_cropped = extra_mask.MaskToImage().mask_to_image(s.params.subject_region.mask_cropped)[0]
        mask_cropped = ImageScaleBy().upscale(mask_cropped, s.params.upscale_method, 1.5)[0]
        mask_cropped = extra_mask.ImageToMask().image_to_mask(mask_cropped, 'red')[0]
        
        latent = VAEEncodeTiled().encode(s.ksampler.vae, inpainted, tile_size=512)[0]
        if s.params.is_model_diffdiff:
            latent = SetLatentNoiseMask().set_mask(latent, mask_cropped)[0]            
            
        latent = KSampler().sample(
            model=s.ksampler.model_inpaint, 
            seed=s.ksampler.seed, 
            steps=s.ksampler.steps, 
            cfg=s.ksampler.cfg, 
            sampler_name=s.ksampler.sampler_name, 
            scheduler=s.ksampler.scheduler, 
            positive=s.ksampler.positive_inpaint, 
            negative=s.ksampler.negative_inpaint, 
            latent_image=latent, 
            denoise=s.ksampler.denoise
        )[0]

        if s.params.is_model_diffdiff:
            latent = RemoveNoiseMask().doit(latent)[0]
        inpainted = VAEDecodeTiled().decode(s.ksampler.vae, latent, tile_size=512)[0]
        s.tile.inpainted = ImageScale().upscale(inpainted, s.params.upscale_method, s.params.mask_region.width, s.params.mask_region.height, "disabled")[0]

    def paste_tile2source(s):
        s.outputs.image = ImagePaste().paste(background_image=s.inputs.source, foreground_image=s.tile.inpainted, x_position=s.params.mask_region.x, y_position=s.params.mask_region.y)[0]

        