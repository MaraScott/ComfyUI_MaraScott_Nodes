#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#       Concatenate up to 24 entries
#
###

from ...inc.profiles.textConcatenate import Node as ProfileNodeTextConcatenate
from ...utils.constants import get_name, get_category

class TextConcatenate_v1:

    NAME = get_name('Text Concatenate', 1, "t")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "hidden": {
                "id":"UNIQUE_ID",
                **ProfileNodeTextConcatenate.ENTRIES,
            },
            "required": {
                "delimiter": ("STRING", { "default": ", ", "label": "Delimiter" }),
            },
            "optional": {
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "fn"
    CATEGORY = get_category('Prompt')

    def fn(self, **kwargs):
        
        delimiter = kwargs.get('delimiter', ", ")
        
        prefix="string"
        filtered_kwargs = {k: v for k, v in kwargs.items() if k.startswith(prefix)}
        strings = list(filtered_kwargs.values())
        concatenated = delimiter.join(strings)

        return (concatenated,)