#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Upscaler Refiner Node.  Upscale and Refine a picture by 2 using a 9 Square Grid to upscale and refine the visual in 9 sequences
#
###

import torch
import comfy
import comfy_extras
import comfy_extras.nodes_custom_sampler
from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
import nodes
import folder_paths

from ...utils.version import VERSION

from ...inc.lib.image import Image

from ...utils.log import *

class UpscalerRefiner_McBoaty:
    
    SIGMAS_TYPES = [
        "BasicScheduler"
        , "SDTurboScheduler"
        , "AlignYourStepsScheduler"
    ]
    
    OUTPUT_NODE = False
    CATEGORY = "MaraScott/upscaling"
    DESCRIPTION = "An \"UPSCALER\" Node"
    FUNCTION = "fn"
    
    @classmethod
    def _get_info(self, image_width, image_height, image_divisible_by_8, output_image_width, output_image_height):
        return [f"""

    IMAGE (INPUT)
        width   :   {image_width}
        height  :   {image_height}
        image divisible by 8 : {image_divisible_by_8}

    ------------------------------

    IMAGE (OUTPUT)
        width   :   {output_image_width}
        height  :   {output_image_height}
        
    NODE INFO
        version : {VERSION}

"""]        
    
    @classmethod    
    def _get_sigmas(self, sigmas_type, model, steps, denoise, scheduler, model_type):
        if sigmas_type == "SDTurboScheduler":
            SigmaScheduler = getattr(comfy_extras.nodes_custom_sampler, sigmas_type)
            sigmas = SigmaScheduler.get_sigmas(SigmaScheduler, model, steps, denoise)[0]
        elif sigmas_type == "AlignYourStepsScheduler":
            SigmaScheduler = AlignYourStepsScheduler
            sigmas = SigmaScheduler.get_sigmas(SigmaScheduler, model_type, steps, denoise)[0]
        else: # BasicScheduler
            SigmaScheduler = getattr(comfy_extras.nodes_custom_sampler, sigmas_type)
            sigmas = SigmaScheduler.get_sigmas(SigmaScheduler, model, scheduler, steps, denoise)[0]

        return sigmas
    
