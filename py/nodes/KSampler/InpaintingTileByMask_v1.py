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

    upscale_method = "lanczos"

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
    # OUTPUT_IS_LIST = (
    #     False,
    #     False,
    #     False,
    #     False,
    #     False,
    # )
    FUNCTION = "fn"
    OUTPUT_NODE = True

    CATEGORY = "MaraScott/Ksampler"

    def fn(s, **kwargs):
        s.extract_and_process_kwargs(**kwargs)
        s.format_inputs_sizes()

        if torch.all(s.colored_mask == 0):
            s.set_default_output()
        else:
            s.set_mask_crop_region()
            s.upscale_inputs_cropped()
            s.set_painted_image_cropped_noised()
            s.set_inpaint_area_noised()
            s.ksample_inpainting()

            s.text_pos_image_inpainted = f"{s.text_pos_image}, {s.text_pos_inpaint}"
            s.text_neg_image_inpainted = s.text_neg_inpaint

            pipe = (
                s.image,
                s.colored_mask,
                s.noise_image,
                s.model_inpaint, 
                s.clip, 
                s.vae, 
                s.text_pos_image_inpainted, 
                s.text_neg_image_inpainted,
                s.seed,
                s.mask_cropped,
                s.image_inpaint_cropped,
                s.width,
                s.height,
                s.x,
                s.y,
                s.inpaint_size,
            )

        return (
            s.image_inpainted,
            s.image_noised,
            s.text_pos_image_inpainted, 
            s.text_neg_image_inpainted,
            pipe
        )

    def extract_and_process_kwargs(s, **kwargs):
        s.image = kwargs.get('image', None)
        s.colored_image = kwargs.get('painted_image', None)
        s.colored_mask = kwargs.get('mask', None)
        s.noise_image = kwargs.get('noise_image', None)

        s.inpaint_size = kwargs.get('inpaint_size', None)
        s.noise_blend = kwargs.get('noise_blend', None)
        s.tile_blend_mode = kwargs.get('tile_blend_mode', None)
        s.tile_opacity = kwargs.get('tile_opacity', None)

        s.model = kwargs.get('model', None)
        s.model_diffdiff = kwargs.get('model_diffdiff', True)
        if s.model_diffdiff:
            s.model_diff = DiffDiff.DifferentialDiffusion().apply(s.model)[0]
            s.model_inpaint = s.model_diff
        else:
            s.model_inpaint = s.model
        s.clip = kwargs.get('clip', None)
        s.vae = kwargs.get('vae', None)

        s.text_pos_image = kwargs.get('text_pos_image', None)
        s.text_pos_inpaint = kwargs.get('text_pos_inpaint', None)
        s.text_neg_inpaint = kwargs.get('text_neg_inpaint', None)
        s.positive_inpaint = CLIPTextEncode().encode(s.clip, s.text_pos_inpaint)[0]
        s.negative_inpaint = CLIPTextEncode().encode(s.clip, s.text_neg_inpaint)[0]

        s.seed = kwargs.get('seed', None)
        s.steps = kwargs.get('steps', None)
        s.cfg = kwargs.get('cfg', None)
        s.sampler_name = kwargs.get('sampler_name', None)
        s.scheduler = kwargs.get('basic_scheduler', None)
        s.denoise = kwargs.get('denoise', None)

    def format_inputs_sizes(s):

        # format and get size from input image
        s.image, s.image_width, s.image_height, s.image_divisible_by_8 = MS_Image().format_2_divby8(s.image)

        # Upscale and process painted image and mask
        s.colored_image = ImageScale().upscale(s.colored_image, s.upscale_method, s.image_width, s.image_height, "center")[0]

        mask_image = extra_mask.MaskToImage().mask_to_image(s.colored_mask)[0]
        mask_image = ImageScale().upscale(mask_image, s.upscale_method, s.image_width, s.image_height, "center")[0]
        s.colored_mask = extra_mask.ImageToMask().image_to_mask(mask_image, 'red')[0]

        s.noise_image = ImageScale().upscale(s.noise_image, s.upscale_method, s.image_width, s.image_height, "center")[0]

    def set_mask_crop_region(s):

        region = WAS_Mask_Crop_Region().mask_crop_region(s.colored_mask, padding=0, region_type="dominant")
        s.mask_cropped = region[0]
        s.x = region[3]
        s.y = region[2]
        s.width = region[6]
        s.height = region[7]

        # image
        s.image_inpaint = extra_images.ImageCrop().crop(s.image, s.width, s.height, s.x, s.y)[0]
        s.painted_image_inpaint = extra_images.ImageCrop().crop(s.colored_image, s.width, s.height, s.x, s.y)[0]
        s.noise_image_inpaint = extra_images.ImageCrop().crop(s.noise_image, s.width, s.height, s.x, s.y)[0]

    def set_painted_image_cropped_noised(s):
        # Blend painted image with noise image
        s.painted_image_noised = WAS_Image_Blend().image_blend(
            image_a=s.painted_image_inpaint, 
            image_b=s.noise_image_inpaint, 
            blend_percentage=s.noise_blend
        )[0]

    def upscale_inputs_cropped(s):
        # Mask Upscale
        mask_cropped_img = extra_mask.MaskToImage().mask_to_image(s.mask_cropped)[0]
        s.image_mask_cropped = ImageScale().upscale(mask_cropped_img, s.upscale_method, s.inpaint_size, s.inpaint_size, "center")[0]
        s.mask_cropped = extra_mask.ImageToMask().image_to_mask(s.image_mask_cropped, 'red')[0]

        # Image Upscale
        s.image_inpaint_cropped = ImageScale().upscale(s.image_inpaint, s.upscale_method, s.inpaint_size, s.inpaint_size, "disabled")[0]

        # Noise Upscale
        s.noise_image_cropped = ImageScale().upscale(s.noise_image, s.upscale_method, s.inpaint_size, s.inpaint_size, "center")[0]

    def set_inpaint_area_noised(s):
        s.image_noised = ImageBlendV2.image_blend_v2(
            ImageBlendV2,
            background_image=s.image_inpaint_cropped, 
            layer_image=s.noise_image_cropped, 
            invert_mask=False, 
            blend_mode=s.tile_blend_mode, 
            opacity=s.tile_opacity, 
            layer_mask=s.mask_cropped
        )[0]

    def ksample_inpainting(s):
        latent_inpaint = VAEEncodeTiled().encode(s.vae, s.image_noised, tile_size=512)[0]

        latent_inpaint = SetLatentNoiseMask().set_mask(latent_inpaint, s.mask_cropped)[0]

        latent_inpainted = KSampler().sample(
            model=s.model_inpaint, 
            seed=s.seed, 
            steps=s.steps, 
            cfg=s.cfg, 
            sampler_name=s.sampler_name, 
            scheduler=s.scheduler, 
            positive=s.positive_inpaint, 
            negative=s.negative_inpaint, 
            latent_image=latent_inpaint, 
            denoise=s.denoise
        )[0]

        latent_inpainted = RemoveNoiseMask().doit(latent_inpainted)[0]
        s.image_inpainted = VAEDecodeTiled().decode(s.vae, latent_inpainted, tile_size=512)[0]

    def set_default_output(s):
        s.image_inpainted = s.image
        s.image_noised = s.noise_image
        s.text_pos_image_inpainted = s.text_pos_image
        s.text_neg_image_inpainted = s.text_neg_inpaint
        s.pipe = (
            s.image,
            s.colored_mask,
            s.noise_image,
            s.model, 
            s.clip, 
            s.vae, 
            s.text_pos_image_inpainted, 
            s.text_neg_image_inpainted,
            s.seed,
            MS_Mask.empty(s.image_width, s.image_height),
            MS_Image.empty(s.image_width, s.image_height),
            s.image_width,
            s.image_height,
            0, 
            0,
            s.inpaint_size
        )


class KSampler_pasteInpaintingTileByMask_v1:

    upscale_method = "lanczos"

    def extract_and_process_kwargs(s, **kwargs):
        s.unique_id = kwargs.get('unique_id', None)
        s.image_tile = kwargs.get('image_tile', None)
        s.mask_tile = kwargs.get('mask_tile', None)
        s.upscale_model_name = kwargs.get('upscale_model', None)
        s.upscale_by_model = kwargs.get('upscale_by_model', None)
        s.subject_opacity = kwargs.get('subject_opacity', None)
        s.ms_pipe = kwargs.get('ms_pipe', None)

        s.image, s.colored_mask, s.noise_image, s.model, s.clip, s.vae, s.text_pos_inpaint, s.text_neg_inpaint, s.seed, s.mask_cropped, s.image_inpaint_cropped, s.width, s.height, s.x, s.y, s.inpaint_size = s.ms_pipe

        s.text_pos_inpaint = kwargs.get('text_pos_inpaint', s.text_pos_inpaint)
        s.text_neg_inpaint = kwargs.get('text_neg_inpaint', s.text_neg_inpaint)

        s.model_inpaint = s.model
        s.positive_inpaint = CLIPTextEncode().encode(s.clip, s.text_pos_inpaint)[0]
        s.negative_inpaint = CLIPTextEncode().encode(s.clip, s.text_neg_inpaint)[0]
        s.seed = kwargs.get('seed', None)
        s.steps = kwargs.get('steps', None)
        s.cfg = kwargs.get('cfg', None)
        s.sampler_name = kwargs.get('sampler_name', None)
        s.scheduler = kwargs.get('basic_scheduler', None)
        s.denoise = kwargs.get('denoise', None)

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
        s.extract_and_process_kwargs(**kwargs)
        s.image, s.image_width, s.image_height, s.image_divisible_by_8 = MS_Image().format_2_divby8(s.image)

        if torch.all(s.colored_mask == 0):
            output_image = s.image
            image = s.image
            text_pos_image_inpainted = s.text_pos_inpaint
            text_neg_image_inpainted = s.text_neg_inpaint
        else:
            region = WAS_Mask_Crop_Region().mask_crop_region(s.mask_tile, padding=0, region_type="dominant")
            x_subject = region[3]
            y_subject = region[2]
            width_subject = region[6]
            height_subject = region[7]

            subject = extra_images.ImageCrop.crop(extra_images.ImageCrop, s.image_tile, width_subject, height_subject, x_subject, y_subject)[0]
            subject_only = ttN_imageREMBG.remove_background(ttN_imageREMBG(), image=subject, image_output="Hide", save_prefix="MaraScott_", prompt=None, extra_pnginfo=None, my_unique_id="{unique_id}")[0]
            subject_only = ImageOpacity.image_opacity(ImageOpacity, image=subject_only, opacity=s.subject_opacity, invert_mask=True)[0]
            subject = ImagePaste.paste(ImagePaste, background_image=s.image_inpaint_cropped, foreground_image=subject_only, x_position=x_subject, y_position=y_subject)[0]
            subject = ImageOpacity.image_opacity(ImageOpacity, image=subject, opacity=100, invert_mask=False)[0]
            subject_upscaled = ImageScale().upscale(subject, s.upscale_method, s.width, s.height, "center")[0]

            output_image = ImagePaste.paste(ImagePaste, background_image=s.image, foreground_image=subject_upscaled, x_position=s.x, y_position=s.y)[0]

            mask_image = extra_mask.MaskToImage().mask_to_image(s.colored_mask)[0]
            mask_image = ImageScale().upscale(mask_image, s.upscale_method, s.width, s.height, "center")[0]

            if s.upscale_by_model:
                upscale_model = extra_upscale_model.UpscaleModelLoader.load_model(extra_upscale_model.UpscaleModelLoader, s.upscale_model_name)[0]
                output_image_upscaled = extra_upscale_model.ImageUpscaleWithModel.upscale(extra_upscale_model.ImageUpscaleWithModel, upscale_model, output_image)[0]
            else:
                output_image_upscaled = ImageScaleBy().upscale(output_image, s.upscale_method, 1.5)[0]
                mask_image = ImageScaleBy().upscale(mask_image, s.upscale_method, 1.5)[0]

            s.colored_mask = extra_mask.ImageToMask().image_to_mask(mask_image, 'red')[0]

            latent = VAEEncodeTiled().encode(s.vae, output_image_upscaled, tile_size=512)[0]
            latent = SetLatentNoiseMask().set_mask(latent, s.colored_mask)[0]

            latent_inpainted = KSampler.sample(
                KSampler,
                model=s.model_inpaint, 
                seed=s.seed, 
                steps=s.steps, 
                cfg=s.cfg, 
                sampler_name=s.sampler_name, 
                scheduler=s.scheduler, 
                positive=s.positive_inpaint, 
                negative=s.negative_inpaint, 
                latent_image=latent, 
                denoise=s.denoise
            )[0]

            latent_inpainted = RemoveNoiseMask().doit(latent_inpainted)[0]
            output_image = VAEDecodeTiled().decode(s.vae, latent_inpainted, tile_size=512)[0]

            output_image = ImageScale().upscale(output_image, s.upscale_method, s.image_width, s.image_height, "center")[0]

            output_image = output_image
            image_inpainted = output_image_upscaled
            image = s.image
            text_pos_image_inpainted = s.text_pos_inpaint
            text_neg_image_inpainted = s.text_neg_inpaint

        return (
            output_image,
            image_inpainted,
            image,
            text_pos_image_inpainted, 
            text_neg_image_inpainted
        )
