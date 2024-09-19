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

# Attempt to import numpy and torch, handle if they are not available
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

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
        log(any)
        value = self.serialize_any(any)
        return {"ui": {"text": (value,)}, "result": (any, )}

    def serialize_any(self, any):
        if any is None:
            return 'None'
        elif isinstance(any, (str, int, float, bool)):
            return str(any)
        elif isinstance(any, (list, tuple)):
            try:
                return json.dumps(any)
            except Exception:
                return str(any)
        elif isinstance(any, dict):
            try:
                return json.dumps(any)
            except Exception:
                return str(any)
        elif np is not None and isinstance(any, np.ndarray):
            return f'Numpy array with shape {any.shape} and dtype {any.dtype}'
        elif torch is not None and isinstance(any, torch.Tensor):
            return f'Torch Tensor with shape {any.shape} and dtype {any.dtype}'
        elif isinstance(any, set):
            return str(any)
        elif isinstance(any, (bytes, bytearray)):
            return f'Bytes/Bytearray of length {len(any)}'
        elif hasattr(any, '__dict__'):
            return f'Object of type {type(any).__name__} with attributes {any.__dict__}'
        else:
            return f'Object of type {type(any).__name__}'
