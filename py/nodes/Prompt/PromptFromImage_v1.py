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

import os
import torch
import numpy as np
from types import SimpleNamespace
from PIL import Image

import folder_paths

from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from ...utils.constants import get_name, get_category
from ...inc.lib.llm import MS_Llm
from ..Util.Model import Common as Common_model
from ...utils.log import log


class PromptFromImage_v1:
    
    NAME = get_name('Prompt from Image')

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
    CATEGORY = get_category('Prompt')

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

class EnhancedPromptFromImage_v1(Common_model):

    NAME = get_name('Enhanced Prompt from Image')

    """
    A node that uses Janus to generate a refined text prompt from an image and an input text prompt.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tuple.
    RETURN_NAMES (`tuple`):
        The name of each output in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    """

    # def __init__(self):

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING",),
                "prompt": ("STRING", {"multiline": True,"default": "Improve the following prompt analyzing the picture which includes :\nobject\nPrompt to modify : \"From raw disorganized data to organized data in a database or multiple databases to beautiful dashboards\"",}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "fn"
    CATEGORY = get_category('Prompt')

    @classmethod
    def fn(self, image_path, prompt):

        # specify the path to the model
        model_path = "deepseek-ai/JanusFlow-1.3B"
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt = MultiModalityCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>\n{prompt}",
                "images": [image_path],
            },
            {"role": "Assistant", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        # # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # # run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        generated_prompt = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

        return (generated_prompt,)
