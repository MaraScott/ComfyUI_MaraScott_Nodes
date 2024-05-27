
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
from nodes import KSampler, CLIPTextEncode, VAEEncodeTiled, VAEDecodeTiled, ImageScale, SetLatentNoiseMask
import folder_paths

from ...vendor.ComfyUI_LayerStyle.py.image_blend_v2 import ImageBlendV2, chop_mode_v2
from ...vendor.ComfyUI_LayerStyle.py.image_opacity import ImageOpacity
from ...vendor.was_node_suite_comfyui.WAS_Node_Suite import WAS_Mask_Crop_Region, WAS_Image_Blend
from ...vendor.ComfyUI_Impact_Pack.modules.impact.util_nodes import RemoveNoiseMask
from ...vendor.mikey_nodes.mikey_nodes import ImagePaste
from ...vendor.ComfyUI_tinyterraNodes.ttNpy.tinyterraNodes import ttN_imageREMBG

from ...utils.log import *
from ...utils.helper import MS_Image, MS_Mask

class KSampler_setInpaintingTileByMask_v1:

    upscale_methos = "lanczos"

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

    def fn(self, **kwargs):

        image = kwargs.get('image', None)
        painted_image = kwargs.get('painted_image', None)
        mask = kwargs.get('mask', None)
        noise_image = kwargs.get('noise_image', None)

        inpaint_size = kwargs.get('inpaint_size', None)
        noise_blend = kwargs.get('noise_blend', None)
        tile_blend_mode = kwargs.get('tile_blend_mode', None)
        tile_opacity = kwargs.get('tile_opacity', None)

        model = kwargs.get('model', None)
        model_diff = DiffDiff.DifferentialDiffusion.apply(DiffDiff.DifferentialDiffusion(), model)[0]
        model_inpaint = model_diff
        clip = kwargs.get('clip', None)
        vae = kwargs.get('vae', None)

        text_pos_image = kwargs.get('text_pos_image', None)
        text_pos_inpaint = kwargs.get('text_pos_inpaint', None)
        text_neg_inpaint = kwargs.get('text_neg_inpaint', None)
        positive_inpaint = CLIPTextEncode.encode(CLIPTextEncode, clip, text_pos_inpaint)[0]
        negative_inpaint = CLIPTextEncode.encode(CLIPTextEncode, clip, text_neg_inpaint)[0]

        seed = kwargs.get('seed', None)
        steps = kwargs.get('steps', None)
        cfg = kwargs.get('cfg', None)
        sampler_name = kwargs.get('sampler_name', None)
        scheduler = kwargs.get('basic_scheduler', None)
        denoise = kwargs.get('denoise', None)

        _image = image
        if _image.shape[0] > 0:
            _image = torch.unsqueeze(_image[0], 0)
        _image = MS_Image.tensor2pil(_image)

        painted_image = ImageScale.upscale(ImageScale, painted_image, self.upscale_methos, _image.width, _image.height, "center")[0]
        mask_image = extra_mask.MaskToImage.mask_to_image(extra_mask.MaskToImage, mask)[0]
        mask_image = ImageScale.upscale(ImageScale, mask_image, self.upscale_methos, _image.width, _image.height, "center")[0]
        mask = extra_mask.ImageToMask.image_to_mask(extra_mask.ImageToMask, mask_image, 'red')[0]
        noise_image = ImageScale.upscale(ImageScale, noise_image, self.upscale_methos, _image.width, _image.height, "center")[0]

        painted_image_noised = WAS_Image_Blend.image_blend(
            WAS_Image_Blend,
            image_a=painted_image, 
            image_b=noise_image, 
            blend_percentage=noise_blend
        )[0]


        if torch.all(mask == 0):
            image_inpainted = image
            image_noised = noise_image
            text_pos_image_inpainted = text_pos_image
            text_neg_image_inpainted = text_neg_inpaint
            pipe = (
                image,
                mask,
                noise_image,
                model, 
                clip, 
                vae, 
                text_pos_image_inpainted, 
                text_neg_image_inpainted,
                seed,
                MS_Mask.empty(_image.width, _image.height),
                MS_Image.empty(_image.width, _image.height),
                _image.width,
                _image.height,
                0, 
                0,
                inpaint_size
            )
        else:
            region = WAS_Mask_Crop_Region.mask_crop_region(WAS_Mask_Crop_Region(), mask, padding=0, region_type="dominant")
            mask_cropped = region[0]
            x = region[3]
            y = region[2]
            width = region[6]
            height = region[7]

            # Mask Upscale
            mask_cropped_img = extra_mask.MaskToImage.mask_to_image(extra_mask.MaskToImage, mask_cropped)[0]
            image_mask_cropped = ImageScale.upscale(ImageScale, mask_cropped_img, self.upscale_methos, inpaint_size, inpaint_size, "center")[0]
            mask_cropped = extra_mask.ImageToMask.image_to_mask(extra_mask.ImageToMask, image_mask_cropped, 'red')[0]

            # Image Upscale
            image_inpaint = extra_images.ImageCrop.crop(extra_images.ImageCrop, image, width, height, x, y)[0]
            image_inpaint_cropped = ImageScale.upscale(ImageScale, image_inpaint, self.upscale_methos, inpaint_size, inpaint_size, "disabled")[0]

            # Noise Upscale
            noise_image = extra_images.ImageCrop.crop(extra_images.ImageCrop, painted_image_noised, width, height, x, y)[0]
            noise_image_cropped = ImageScale.upscale(ImageScale, noise_image, self.upscale_methos, inpaint_size, inpaint_size, "center")[0]

            image_noised = ImageBlendV2.image_blend_v2(
                ImageBlendV2,
                background_image=image_inpaint_cropped, 
                layer_image=noise_image_cropped, 
                invert_mask=False, 
                blend_mode=tile_blend_mode, 
                opacity=tile_opacity, 
                layer_mask=mask_cropped
            )[0]

            latent_inpaint = VAEEncodeTiled.encode(VAEEncodeTiled, vae, image_noised, tile_size=512)[0]

            latent_inpaint = SetLatentNoiseMask.set_mask(SetLatentNoiseMask, latent_inpaint, mask_cropped)[0]

            latent_inpainted = KSampler.sample(
                KSampler,
                model=model_inpaint, 
                seed=seed, 
                steps=steps, 
                cfg=cfg, 
                sampler_name=sampler_name, 
                scheduler=scheduler, 
                positive=positive_inpaint, 
                negative=negative_inpaint, 
                latent_image=latent_inpaint, 
                denoise=denoise
            )[0]

            latent_inpainted = RemoveNoiseMask.doit(RemoveNoiseMask, latent_inpainted)[0]

            image_inpainted = VAEDecodeTiled.decode(VAEDecodeTiled, vae, latent_inpainted, tile_size=512)[0]

            text_pos_image_inpainted = f"{text_pos_image}, {text_pos_inpaint}"
            text_neg_image_inpainted = text_neg_inpaint

            pipe = (
                image,
                mask,
                noise_image,
                model, 
                clip, 
                vae, 
                text_pos_image_inpainted, 
                text_neg_image_inpainted,
                seed,
                mask_cropped,
                image_inpaint_cropped,
                width,
                height,
                x,
                y,
                inpaint_size,
            )

        return (
            image_inpainted,
            image_noised,
            text_pos_image_inpainted, 
            text_neg_image_inpainted,
            pipe
        )
    
class KSampler_pasteInpaintingTileByMask_v1:

    upscale_methos = "lanczos"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_tile": ("IMAGE", { "label": "Image (Tile)" }),
                "mask_tile": ("MASK", { "label": "Mask (Tile)" }),
                "upscale_model": (folder_paths.get_filename_list("upscale_models"),),
                "ms_pipe": ("MS_INPAINTINGTILEBYMASK_PIPE", { "label": "pipe (InpaintingTileByMask)" }),

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

    # INPUT_IS_LIST = False
    RETURN_TYPES = (
        'IMAGE',
        'IMAGE',
        'STRING',
        'STRING',
    )
    RETURN_NAMES = (
        'output_image',
        'image',
        'text_pos_image_inpainted',
        'text_neg_image_inpainted',
    )
    RETURN_LABELS = (
        'Inpainted Image',
        'Inpainted Tile',
        'positive (text)',
        'negative (text)',
    )
    # OUTPUT_IS_LIST = (
    #     False,
    #     False,
    #     False,
    #     False,
    # )
    FUNCTION = "fn"
    OUTPUT_NODE = True

    CATEGORY = "MaraScott/Ksampler"

    def fn(self, **kwargs):

        unique_id = kwargs.get('unique_id', None)
        image_tile = kwargs.get('image_tile', None)
        mask_tile = kwargs.get('mask_tile', None)
        upscale_model_name = kwargs.get('upscale_model', None)
        subject_opacity = kwargs.get('subject_opacity', None)
        ms_pipe = kwargs.get('ms_pipe', None)

        image, mask, noise_image, model, clip, vae, text_pos_inpaint, text_neg_inpaint, seed, mask_cropped, image_inpaint_cropped, width, height, x, y, inpaint_size = ms_pipe

        model_diff = DiffDiff.DifferentialDiffusion.apply(DiffDiff.DifferentialDiffusion(), model)[0]
        model_inpaint = model_diff
        positive_inpaint = CLIPTextEncode.encode(CLIPTextEncode, clip, text_pos_inpaint)[0]
        negative_inpaint = CLIPTextEncode.encode(CLIPTextEncode, clip, text_neg_inpaint)[0]
        seed = kwargs.get('seed', None)
        steps = kwargs.get('steps', None)
        cfg = kwargs.get('cfg', None)
        sampler_name = kwargs.get('sampler_name', None)
        scheduler = kwargs.get('basic_scheduler', None)
        denoise = kwargs.get('denoise', None)

        if torch.all(mask == 0):
            output_image = image
            image = image
            text_pos_image_inpainted = text_pos_inpaint
            text_neg_image_inpainted = text_neg_inpaint
        else:
            region = WAS_Mask_Crop_Region.mask_crop_region(WAS_Mask_Crop_Region(), mask_tile, padding=0, region_type="dominant")
            x_subject = region[3]
            y_subject = region[2]
            width_subject = region[6]
            height_subject = region[7]

            subject = extra_images.ImageCrop.crop(extra_images.ImageCrop, image_tile, width_subject, height_subject, x_subject, y_subject)[0]
            subject_only = ttN_imageREMBG.remove_background(ttN_imageREMBG(), image=subject, image_output="Hide", save_prefix="MaraScott_", prompt=None, extra_pnginfo=None, my_unique_id="{unique_id}")[0]
            subject_only = ImageOpacity.image_opacity(ImageOpacity, image=subject_only, opacity=subject_opacity, invert_mask=True)[0]
            subject = ImagePaste.paste(ImagePaste,background_image=image_inpaint_cropped, foreground_image=subject_only, x_position=x_subject, y_position=y_subject)[0]
            subject = ImageOpacity.image_opacity(ImageOpacity, image=subject, opacity=100, invert_mask=False)[0]
            subject_upscaled = ImageScale.upscale(ImageScale, subject, self.upscale_methos, width, height, "center")[0]
            output_image = ImagePaste.paste(ImagePaste,background_image=image, foreground_image=subject_upscaled, x_position=x, y_position=y)[0]
            upscale_model = extra_upscale_model.UpscaleModelLoader.load_model(extra_upscale_model.UpscaleModelLoader, upscale_model_name)[0]
            output_image_upscaled = extra_upscale_model.ImageUpscaleWithModel.upscale(extra_upscale_model.ImageUpscaleWithModel, upscale_model, output_image)[0]
            mask_image = extra_mask.MaskToImage.mask_to_image(extra_mask.MaskToImage, mask)[0]
            mask_image = ImageScale.upscale(ImageScale, mask_image, self.upscale_methos, width, height, "center")[0]
            mask = extra_mask.ImageToMask.image_to_mask(extra_mask.ImageToMask, mask_image, 'red')[0]
            

            latent = VAEEncodeTiled.encode(VAEEncodeTiled, vae, output_image_upscaled, tile_size=512)[0]
            latent = SetLatentNoiseMask.set_mask(SetLatentNoiseMask, latent, mask)[0]

            latent_inpainted = KSampler.sample(
                KSampler,
                model=model_inpaint, 
                seed=seed, 
                steps=steps, 
                cfg=cfg, 
                sampler_name=sampler_name, 
                scheduler=scheduler, 
                positive=positive_inpaint, 
                negative=negative_inpaint, 
                latent_image=latent, 
                denoise=denoise
            )[0]

            latent_inpainted = RemoveNoiseMask.doit(RemoveNoiseMask, latent_inpainted)[0]
            output_image = VAEDecodeTiled.decode(VAEDecodeTiled, vae, latent_inpainted, tile_size=512)[0]


            output_image = output_image
            image = image
            text_pos_image_inpainted = text_pos_inpaint
            text_neg_image_inpainted = text_neg_inpaint

        return (
            output_image,
            image,
            text_pos_image_inpainted, 
            text_neg_image_inpainted
        )    