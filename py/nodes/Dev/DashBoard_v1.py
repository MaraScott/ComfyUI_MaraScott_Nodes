#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#       DashBoard
#
###

from ...utils.constants import get_name, get_category
from ...utils.helper import AnyType

any_type = AnyType("*")

class DashBoard_v1:

    NAME = get_name('DashBoard')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "requeue": ("INT", { "label": "requeue (automatic or manual)", "default": 0, "min": 0, "max": 99999999999, "step": 1}),
                "any": (any_type, {}),
            },
            "optional": {
            },
            "hidden": {
                "id":"UNIQUE_ID"
            },
        }

    RETURN_TYPES = (
        any_type,
    )
    
    RETURN_NAMES = (
        "any",
    )
    
    OUTPUT_IS_LIST = (
        False,
    )
    
    OUTPUT_NODE = False
    CATEGORY = get_category('Dev')
    DESCRIPTION = "DashBoard"
    FUNCTION = "fn"
    
    def fn(self, any=None, requeue=0, id=None):
        value = "Please Use AnyBus_v2 to activate this node"
        return {"ui": {"text": (value,)}, "result": (any, )}