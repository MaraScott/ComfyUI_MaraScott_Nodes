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

    def fn(self, clip, text, model_options={}):
        
        clip_l_class = model_options.get("clip_l_class", SDClipModel)
        
        model = clip_l_class(
            layer="last", 
            layer_idx=None, 
            device="cpu", 
            dtype=None, 
            layer_norm_hidden_state=True, 
            return_projected_pooled=True, 
            model_options=model_options
        )

        # Example input tokens (this would usually come from a tokenizer)
        conditioning = CLIPTextEncode.encode(CLIPTextEncode, clip, text)
        hidden_states = [vars(model), dir(model)]

        # Print hidden states
        return (conditioning, hidden_states)