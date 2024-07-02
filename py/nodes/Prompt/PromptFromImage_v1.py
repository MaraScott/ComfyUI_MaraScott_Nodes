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

from ...inc.lib.llm import MS_Llm

from ...utils.log import log

class PromptFromImage_v1:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"label": "image"}),
                "vision_llm_model": (MS_Llm.VISION_LLM_MODELS, { "label": "Vision LLM Model", "default": "microsoft/Florence-2-large" }),
                "llm_model": (MS_Llm.LLM_MODELS, { "label": "LLM Model", "default": "llama3-70b-8192" }),
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
        self.LLM = SimpleNamespace(
            vision_model_name = kwargs.get('vision_llm_model', None),
            model_name = kwargs.get('llm_model', None),
            model = None,
        )
        self.LLM.model = MS_Llm(self.LLM.vision_model_name, self.LLM.model_name)

        self.OUPUTS = SimpleNamespace(
            prompt = self.LLM.model.vision_llm.generate_prompt(self.INPUTS.image)
        )
            
        return {"ui": {"text": self.OUPUTS.prompt}, "result": (self.OUPUTS.prompt,)}