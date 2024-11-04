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
import re
import json
import numpy as np
import torch
from collections import defaultdict
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
        value = self.serialize_any(any)
        return {"ui": {"text": (value,)}, "result": (any, )}

    def serialize_any(self, any):
        if self.is_user_defined_object(any):
            try:
                analyzed_object = self.analyze_object(any)
                analyzed_object_info = {
                    "state_dict": self.get_max_block_indexes(any.model_state_dict()),
                    "analyzed_object": analyzed_object,
                    "input": json.dumps(any)
                }
                return json.dumps(analyzed_object_info, indent=2, default=str).replace('\\n', '\n')
            except Exception:
                return str(any)
        elif any is None:
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
        
    def is_user_defined_object(self, obj):
        # Check if the object is an instance of a class but not a built-in type
        return not isinstance(obj, (int, float, str, list, dict, tuple, set, bool, type(None)))

    def analyze_object(self, obj):
        properties = []
        methods = []
        
        for attribute_name in dir(obj):
            attribute = getattr(obj, attribute_name)
            
            # Check if the attribute is callable (method)
            if callable(attribute):
                methods.append(attribute_name)
            else:
                properties.append(attribute_name)
        
        return {"properties": properties, "methods": methods}
    
    def display_object_info(self, info):
        # Extract properties and methods
        properties = info.get("properties", [])
        methods = info.get("methods", [])
        
        # Organize properties and methods
        dunder_methods = [m for m in methods if m.startswith("__") and m.endswith("__")]
        user_defined_methods = [m for m in methods if not (m.startswith("__") and m.endswith("__"))]
        user_defined_properties = [p for p in properties if not (p.startswith("__") and p.endswith("__"))]
        
        # Create the display structure
        return {
            "Dunder Methods": dunder_methods,
            "User-defined Properties": user_defined_properties,
            "User-defined Methods": user_defined_methods
        }
        
    def get_max_block_indexes(self, data):
        max_indexes = defaultdict(int)
        # pattern = re.compile(r"(\w*block\w*\.\w+)\.(\d+)")
        pattern = re.compile(r"(\w+\.\w+)\.(\d+)")
        for key in data.keys():
            if "block" in key: 
                match = pattern.search(key)
                if match:
                    block_prefix = match.group(1)
                    index = int(match.group(2))
                    if block_prefix.startswith("diffusion_model."):
                        block_prefix = block_prefix.replace("diffusion_model.", "", 1)
                    max_indexes[block_prefix] = max(max_indexes[block_prefix], index)
        
        return max_indexes
        