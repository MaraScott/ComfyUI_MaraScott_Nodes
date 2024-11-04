#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

import re
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

class TautologyStr(str):
    def __ne__(self, other):
        return False

class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index>0:
            index=0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return TautologyStr(item)
        return item

def is_user_defined_object(obj):
    # Check if the object is an instance of a class but not a built-in type
    return not isinstance(obj, (int, float, str, list, dict, tuple, set, bool, type(None)))

def analyze_object(obj):
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

def natural_key(entry):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', entry)]