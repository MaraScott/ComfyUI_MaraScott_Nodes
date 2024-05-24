
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

import comfy
from comfy_extras import nodes_differential_diffusion as DiffDiff
from nodes import KSampler, CLIPTextEncode, VAEEncodeTiled, VAEDecodeTiled, ImageScale
import folder_paths

from ...vendor.ComfyUI_LayerStyle.py.image_blend_v2 import ImageBlendV2

from ...utils.log import *

class KSampler_InpaintingTileByMask_v1:

    upscale_methos = "lanczos"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", { "label": "Image" }),
                "mask": ("MASK", { "label": "Mask" }),
                "noise_image": ("IMAGE", { "label": "Image (Noise)" }),
                "noise_opacity": ("INT", { "label": "Opacity (Noise)", "default": 19, "min": 0, "max": 100, "step": 1 }),
                "model": ("MODEL", { "label": "Model" }),
                "clip": ("CLIP", { "label": "Clip" }),
                "vae": ("VAE", { "label": "VAE" }),

                "text_pos_image": ("STRING", { "label": "Positive (text img)", "forceInput": True }),
                "text_pos_inpaint": ("STRING", { "label": "Positive (text)", "forceInput": True }),
                "text_neg_inpaint": ("STRING", { "label": "Negative (text)", "forceInput": True }),

                "upscale_model": (folder_paths.get_filename_list("upscale_models"),),

                "inpaint_size": ("INT", { "label": "Inpaint Size", "default": 1024, "min": 512, "max": 1024, "step": 512}),

                "inpaint_rescale": ("FLOAT", { "label": "Rescale factor", "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),

                "seed": ("INT", { "label": "Seed", "default": 4, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", { "label": "Steps", "default": 10, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", { "label": "CFG", "default": 8, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, { "label": "Sampler Name" }),
                "basic_scheduler": (comfy.samplers.KSampler.SCHEDULERS, { "label": "Basic Scheduler" }),
                "denoise": ("FLOAT", { "label": "Denoise", "default": 0.51, "min": 0.0, "max": 1.0, "step": 0.01}),

                "inpaint_treshold": ("INT", { "label": "Mask Treshold", "default": 200, "min": 0, "max": 255, "step": 1}),

            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    INPUT_IS_LIST = False
    RETURN_TYPES = (
        'IMAGE',
        'STRING',
        'STRING',
    )
    RETURN_NAMES = (
        'image_inpainted',
        'text_pos_image_inpainted',
        'text_neg_image_inpainted',
    )
    RETURN_LABELS = (
        'Image',
        'positive (text)',
        'negative (text)',
    )
    FUNCTION = "fn"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,)

    CATEGORY = "MaraScott/Ksampler"

    def fn(self, **kwargs):

        image = kwargs.get('image', None)
        mask = kwargs.get('mask', None)
        model = kwargs.get('model', None)
        model_diff = DiffDiff.apply(model)
        model_inpaint = model_diff
        clip = DiffDiff.apply(clip)
        vae = DiffDiff.apply(vae)
        text_pos_image = kwargs.get('text_pos_image', None)
        text_pos_inpaint = kwargs.get('text_pos_inpaint', None)
        text_neg_inpaint = kwargs.get('text_neg_inpaint', None)
        positive_inpaint = CLIPTextEncode.encode(clip, text_pos_inpaint)
        negative_inpaint = CLIPTextEncode.encode(clip, text_neg_inpaint)
        seed = kwargs.get('seed', None)
        steps = kwargs.get('steps', None)
        cfg = kwargs.get('cfg', None)
        sampler_name = kwargs.get('sampler_name', None)
        scheduler = kwargs.get('basic_scheduler', None)
        denoise = kwargs.get('denoise', None)
        noise_image = kwargs.get('noise_image', None)
        inpaint_size = kwargs.get('inpaint_size', None)
        noise_opacity = kwargs.get('noise_opacity', None)
        
        image = ImageScale.upscale(image, self.upscale_methos, inpaint_size, inpaint_size, "disabled")
        noise_image = ImageScale.upscale(noise_image, self.upscale_methos, inpaint_size, inpaint_size, "center")

        image_inpaint = ImageBlendV2.image_blend_v2(
            background_image=image, 
            layer_image=noise_image, 
            invert_mask=False, 
            blend_mode="overlay", 
            opacity=noise_opacity, 
            layer_mask=mask
        )

        latent_image = VAEEncodeTiled.encode(vae, image_inpaint)

        latent_inpaint = KSampler.sample(
            self, 
            model_inpaint, 
            seed, 
            steps, 
            cfg, 
            sampler_name, 
            scheduler, 
            positive_inpaint, 
            negative_inpaint, 
            latent_image, 
            denoise
        )

        image_inpainted = image
        text_pos_image_inpainted = f"{text_pos_image}, {text_pos_inpaint}"
        text_neg_image_inpainted = text_neg_inpaint

        return (
            image_inpainted, 
            text_pos_image_inpainted, 
            text_neg_image_inpainted
        )