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
import json
from ...utils.helper import AlwaysEqualProxy, AnyType
from ...utils.constants import get_name, get_category
from ...utils.log import log

any_type = AnyType("*")

class DisplayInfo_v2:
    
    NAME = get_name('Display Any')
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = False
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "fn"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,)

    CATEGORY = get_category('utils')

    def fn(self, any=None, unique_id=None, extra_pnginfo=None):

        value = 'None'        
        if any is not None:
            try:
                value = json.dumps(any)
            except Exception:
                try:
                    value = str(any)
                except Exception:
                    value = 'any exists, but could not be serialized.'
                    
        return {"ui": {"text": (value,)}, "result": (value,)}
