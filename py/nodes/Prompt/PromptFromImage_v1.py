#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#       Display Info or any string
#
#       Largely inspired by PYSSSSS - ShowText 
#
###

from types import SimpleNamespace

from ...inc.lib.image import MS_Image_v2 as MS_Image
# from ...inc.lib.llm~ import MS_Llm
from ...vendor.ComfyUI_WD14_Tagger.wd14tagger import wait_for_async, tag

from ...utils.log import *

class PromptFromImage_v1:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"label": "image"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = False
    FUNCTION = "fn"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,)
    CATEGORY = "MaraScott/Prompt"

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "Prompt",
    )
    
    @classmethod
    def fn(self, **kwargs):
        
        self.INPUTS = SimpleNamespace(
            image = kwargs.get('image', None)
        )
        # self.LLM = SimpleNamespace(
        #     vision_model_name = kwargs.get('vision_llm_model', None),
        #     model_name = kwargs.get('llm_model', None),
        #     model = None,
        # )
        # self.LLM.model = MS_Llm(self.LLM.vision_model_name, self.LLM.model_name)

        self.OUPUTS = SimpleNamespace(
            prompt = wait_for_async(lambda: tag(MS_Image.tensor2pil(self.INPUTS.image), model_name="wd-v1-4-moat-tagger-v2", threshold=0.35, character_threshold=0.85, exclude_tags="", replace_underscore=False, trailing_comma=False))
        )
            
        return {"ui": {"text": self.OUPUTS.prompt}, "result": (self.OUPUTS.prompt,)}