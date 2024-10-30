#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#       Text conversion in other format
#
###

from ...utils.constants import get_name, get_category

class TextConversion_StringToList_v1:

    NAME = get_name('String To List')

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"multiline": True})
        }}
    RETURN_NAMES = ("list", "text (original)", "lines count")
    RETURN_TYPES = ("STRING", "STRING", "INT")
    OUTPUT_IS_LIST = (True, False, False)
    
    FUNCTION = "fn"
    CATEGORY = get_category('Prompt')

    def fn(self, **kwargs):
        
        text = kwargs.get('text', None)
        list = []
        if text is not None:
            lines = text.splitlines()
            list = [line.strip() for line in lines if line.strip()]
        count = len(list)
        return ([list], text, count, )