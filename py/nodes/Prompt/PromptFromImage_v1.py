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
from transformers import AutoProcessor, AutoModelForCausalLM

import folder_paths

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
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True,"default": "Improve the following prompt analyzing the picture which includes :\nobject\nPrompt to modify : \"From raw disorganized data to organized data in a database or multiple databases to beautiful dashboards\"",}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "fn"
    CATEGORY = get_category('Prompt')

    @classmethod
    def fn(self, image, prompt):

        model_path = os.path.join(folder_paths.get_folder_paths("LLM")[0],'deepseek-ai-JanusFlow-1.3B.safetensors')

        self.download_large_file(url='https://cdn-lfs-us-1.hf.co/repos/7a/2b/7a2ba2c22c7681f280799868e1e88333bcb081fbf6f240f449ae07bbda72db82/2e43167cf7a74edfaed1c35e7450c0e1f8345ff39f825e5372a0dbda8ef77eeb?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27model.safetensors%3B+filename%3D%22model.safetensors%22%3B&Expires=1732829193&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczMjgyOTE5M319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzdhLzJiLzdhMmJhMmMyMmM3NjgxZjI4MDc5OTg2OGUxZTg4MzMzYmNiMDgxZmJmNmYyNDBmNDQ5YWUwN2JiZGE3MmRiODIvMmU0MzE2N2NmN2E3NGVkZmFlZDFjMzVlNzQ1MGMwZTFmODM0NWZmMzlmODI1ZTUzNzJhMGRiZGE4ZWY3N2VlYj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=MofvVcjL3xmRt8uQSIO71Ow0Pi9fdF1z1o%7Er92I7H4Qbp%7EGt%7EcIBkdndu%7El1glidUuq7%7ExUsi9Ah-T3pyY1Yn7Qgg-Ol6QguAW6tDwNzRNgfyIQM7LxufoiUuH5ENszFJrSSC-FUFSZYIifZq-lxkDSx26zB9Puzjyef%7EdVpIyWy3OIaPZ5rjjMSGmIn1tsBBR6VsGNS8Mn%7E3p8mC5e7BaNFi-r-gaJ8HZWGmW7VRqQH0d8qVFJvWsEn6NU%7EOteFL3u5BqE8BXJxkQqULbDr8PJ-5u8mQuXF0ijFM9rwcWPbYwTfMn%7EfL2ahcSpKZDmDHVHHrghQfjWmytGbWOmCFQ__&Key-Pair-Id=K24J24Z295AEI9', filename=model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Convert the input image to a PIL Image if it's a tensor or NumPy array
        if isinstance(image, torch.Tensor):
            # Convert tensor to NumPy array
            image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))

        # Step 1: Prepare inputs for Janus
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        # Step 2: Generate the prompt using Janus
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=150,
            num_beams=5,
            early_stopping=True
        )

        # Decode the generated tokens
        generated_prompt = self.processor.decode(outputs[0], skip_special_tokens=True)

        return (generated_prompt,)
