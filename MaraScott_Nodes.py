#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from . import __SESSIONS_DIR__, __PROFILES_DIR__
from .py.nodes.AnyBusNode import AnyBusNode as BusNode_v1
from .py.nodes.AnyBusNode import AnyBusNode as AnyBusNode_V1
from .py.nodes.AnyBusNode import AnyBusNode as AnyBusNode_V2
from .py.nodes.DisplayInfoNode import DisplayInfoNode as DisplayInfoNode_v1
from .py.nodes.UpscalerRefiner.McBoaty_v1 import UpscalerRefiner_McBoaty_v1
from .py.nodes.UpscalerRefiner.McBoaty_v2 import UpscalerRefiner_McBoaty_v2

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MaraScottAnyBusNode": AnyBusNode_V2,
    "MaraScottDisplayInfoNode": DisplayInfoNode_v1,
    "MaraScottUpscalerRefinerNode_v2": UpscalerRefiner_McBoaty_v2,

    "MarasitBusNode": BusNode_v1,
    "MarasitAnyBusNode": AnyBusNode_V1,
    "MarasitDisplayInfoNode": DisplayInfoNode_v1,
    "MarasitUpscalerRefinerNode": UpscalerRefiner_McBoaty_v1,
    "MarasitUpscalerRefinerNode_v2": UpscalerRefiner_McBoaty_v2,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes 
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaraScottAnyBusNode": "\ud83d\udc30 AnyBus - UniversalBus v2 /*",
    "MaraScottDisplayInfoNode": "\ud83d\udc30 Display Info - Text v1 /i",
    "MaraScottUpscalerRefinerNode_v2": "\ud83d\udc30 Large Refiner - McBoaty v2 /u",

    "MarasitBusNode": "\ud83d\udc30 Bus Node v1 (deprecated) /*",
    "MarasitAnyBusNode": "\ud83d\udc30 AnyBus - UniversalBus v1 (deprecated) /*",
    "MarasitDisplayInfoNode": "\ud83d\udc30 Display Info - Text v1 (deprecated) /i",
    "MarasitUpscalerRefinerNode": "\ud83d\udc30 Large Refiner - McBoaty v1 (deprecated)/u",
    "MarasitUpscalerRefinerNode_v2": "\ud83d\udc30 Large Refiner - McBoaty v2 (deprecated)/u",
}

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')
