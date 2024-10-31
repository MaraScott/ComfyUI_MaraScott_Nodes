#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#       Display Info or any string
#
#
###

import comfy
from nodes import CLIPTextEncode

from comfy.sd1_clip import SDClipModel 

class CLIPTextEncodeWithLayerAssociation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "dynamicPrompts": True
                }), 
                "clip": ("CLIP", )
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )

    RETURN_NAMES = (
        "CONDITIONING", 
        "Hidden States", 
    )

    CATEGORY = "MaraScott/utils"
    DESCRIPTION = "A CLIP Layer Text Detailer Node"    
    FUNCTION = "fn"

    def fn(self, clip, text):

        # Usage example
        # Initialize the custom model with necessary parameters
        model = SDClipModel(
            version="openai/clip-vit-large-patch14",
            device="cpu",
            max_length=77,
            freeze=True,
            layer="last",
            layer_idx=None,
            textmodel_json_config=None,
            dtype=None,
            model_class=comfy.clip_model.CLIPTextModel,
            special_tokens={"start": 49406, "end": 49407, "pad": 49407},
            layer_norm_hidden_state=True,
            enable_attention_masks=False,
            return_projected_pooled=True
        )

        # Example input tokens (this would usually come from a tokenizer)
        conditioning = CLIPTextEncode.encode(CLIPTextEncode, clip, text)
        hidden_states = model.get_hidden_states(text)

        # Print hidden states
        return (conditioning, hidden_states)