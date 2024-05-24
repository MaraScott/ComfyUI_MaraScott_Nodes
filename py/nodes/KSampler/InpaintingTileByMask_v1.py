
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
import folder_paths

from ...utils.log import *

class KSampler_InpaintingTileByMask_v1:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", { "label": "Image" }),
                "mask": ("MASK", { "label": "Mask" }),
                "noise_image": ("IMAGE", { "label": "Image (Noise)" }),
                "noise_opacity": ("IMAGE", { "label": "Opacity (Noise)", "default": 19, "min": 0, "max": 100, "step": 1 }),
                "model": ("MODEL", { "label": "Model" }),
                "clip": ("CLIP", { "label": "Clip" }),
                "vae": ("VAE", { "label": "VAE" }),

                "text_pos_image": ("STRING", { "label": "Positive (text img)", "forceInput": True }),
                "text_pos_inpaint": ("STRING", { "label": "Positive (text)", "forceInput": True }),
                "text_neg_inpaint": ("STRING", { "label": "Negative (text)", "forceInput": True }),

                "upscale_model": (folder_paths.get_filename_list("upscale_models"),),

                "inpaint_size": ("INT", { "label": "Inpaint Size", "default": 1024, "min": 512, "max": 1024, "step": 512}),

                "inpaint_rescale": ("FLOAT", { "label": "Rescale factor", "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),

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
        text_pos_image = kwargs.get('text_pos_image', None)
        text_pos_inpaint = kwargs.get('text_pos_inpaint', None)
        text_neg_inpaint = kwargs.get('text_neg_inpaint', None)

        image_inpainted = image
        text_pos_image_inpainted = f"{text_pos_image}, {text_pos_inpaint}"
        text_neg_image_inpainted = text_neg_inpaint

        return (
            image_inpainted, 
            text_pos_image_inpainted, 
            text_neg_image_inpainted
        )