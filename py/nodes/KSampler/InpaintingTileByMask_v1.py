
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
from comfy_extras import nodes_differential_diffusion as DiffDiff, nodes_images as extra_images, nodes_mask as extra_mask
from nodes import KSampler, CLIPTextEncode, VAEEncodeTiled, VAEDecodeTiled, ImageScale, SetLatentNoiseMask
import folder_paths

from ...vendor.ComfyUI_LayerStyle.py.image_blend_v2 import ImageBlendV2
from ...vendor.was_node_suite_comfyui.WAS_Node_Suite import WAS_Mask_Crop_Region
from ...vendor.ComfyUI_Impact_Pack.modules.impact.util_nodes import RemoveNoiseMask

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
        model_diff = DiffDiff.DifferentialDiffusion.apply(DiffDiff.DifferentialDiffusion, model)
        model_inpaint = model_diff
        clip = kwargs.get('clip', None)
        vae = kwargs.get('vae', None)
        text_pos_image = kwargs.get('text_pos_image', None)
        text_pos_inpaint = kwargs.get('text_pos_inpaint', None)
        text_neg_inpaint = kwargs.get('text_neg_inpaint', None)
        positive_inpaint = CLIPTextEncode.encode(CLIPTextEncode, clip, text_pos_inpaint)
        negative_inpaint = CLIPTextEncode.encode(CLIPTextEncode, clip, text_neg_inpaint)
        seed = kwargs.get('seed', None)
        steps = kwargs.get('steps', None)
        cfg = kwargs.get('cfg', None)
        sampler_name = kwargs.get('sampler_name', None)
        scheduler = kwargs.get('basic_scheduler', None)
        denoise = kwargs.get('denoise', None)
        noise_image = kwargs.get('noise_image', None)
        inpaint_size = kwargs.get('inpaint_size', None)
        noise_opacity = kwargs.get('noise_opacity', None)

        region_tensor, top, left, width, height, _ = WAS_Mask_Crop_Region.mask_crop_region(WAS_Mask_Crop_Region, mask, padding=0, region_type="dominant")
        mask_cropped = region_tensor
        x = left
        y = top

        # Mask Upscale
        mask_cropped_img = extra_mask.MaskToImage.mask_to_image(extra_mask.MaskToImage, mask)
        image_mask_cropped = ImageScale.upscale(ImageScale, mask_cropped_img, self.upscale_methos, inpaint_size, inpaint_size, "disabled")
        mask_cropped = extra_mask.ImageToMask.image_to_mask(extra_mask.ImageToMask, image_mask_cropped)

        # Image Upscale
        image_inpaint = extra_images.ImageCrop.crop(extra_images.ImageCrop, image, width, height, x, y)
        image_inpaint_cropped = ImageScale.upscale(ImageScale, image_inpaint, self.upscale_methos, inpaint_size, inpaint_size, "disabled")

        # Noise Upscale
        noise_image = ImageScale.upscale(ImageScale, noise_image, self.upscale_methos, inpaint_size, inpaint_size, "center")

        image_inpaint = ImageBlendV2.image_blend_v2(
            ImageBlendV2,
            background_image=image_inpaint_cropped, 
            layer_image=noise_image, 
            invert_mask=False, 
            blend_mode="overlay", 
            opacity=noise_opacity, 
            layer_mask=mask_cropped
        )

        latent_inpaint = VAEEncodeTiled.encode(VAEEncodeTiled, vae, image_inpaint, tile_size=512)

        latent_inpaint = SetLatentNoiseMask.set_mask(SetLatentNoiseMask, latent_inpaint, mask_cropped)

        latent_inpainted = KSampler.sample(
            KSampler,
            self, 
            model_inpaint, 
            seed, 
            steps, 
            cfg, 
            sampler_name, 
            scheduler, 
            positive_inpaint, 
            negative_inpaint, 
            latent_inpaint, 
            denoise
        )

        latent_inpainted = RemoveNoiseMask.doit(RemoveNoiseMask, latent_inpainted)

        image_inpainted = VAEDecodeTiled.decode(VAEDecodeTiled, vae, latent_inpainted, tile_size=512)


        text_pos_image_inpainted = f"{text_pos_image}, {text_pos_inpaint}"
        text_neg_image_inpainted = text_neg_inpaint

        return (
            image_inpainted, 
            text_pos_image_inpainted, 
            text_neg_image_inpainted
        )