#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from .py.nodes.AnyBus.Bus_v2 import AnyBusNode as BusNode_v1
from .py.nodes.AnyBus.Bus_v2 import AnyBusNode as AnyBusNode_V1
from .py.nodes.AnyBus.Bus_v2 import AnyBusNode as AnyBusNode_V2
from .py.nodes.AnyBus.Bus_v3 import AnyBusNode as AnyBusNode_V3
from .py.nodes.DisplayInfoNode import DisplayInfoNode as DisplayInfoNode_v1
from .py.nodes.UpscalerRefiner.McBoaty_v1 import UpscalerRefiner_McBoaty_v1
from .py.nodes.UpscalerRefiner.McBoaty_v2 import UpscalerRefiner_McBoaty_v2

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MaraScottAnyBusNode_v3": AnyBusNode_V3,
    "MaraScottAnyBusNode": AnyBusNode_V2,
    "MaraScottDisplayInfoNode": DisplayInfoNode_v1,
    "MaraScottUpscalerRefinerNode_v2": UpscalerRefiner_McBoaty_v2,

    "MarasitBusNode": BusNode_v1,
    "MarasitUniversalBusNode": BusNode_v1,
    "MarasitAnyBusNode": AnyBusNode_V1,
    "MarasitDisplayInfoNode": DisplayInfoNode_v1,
    "MarasitUpscalerRefinerNode": UpscalerRefiner_McBoaty_v1,
    "MarasitUpscalerRefinerNode_v2": UpscalerRefiner_McBoaty_v2,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes 
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaraScottAnyBusNode_v3": "\ud83d\udc30 AnyBus - UniversalBus v3 /*",
    "MaraScottDisplayInfoNode": "\ud83d\udc30 Display Info - Text v1 /i",
    "MaraScottUpscalerRefinerNode_v2": "\ud83d\udc30 Large Refiner - McBoaty v2 /u",

    "MarasitBusNode": "\u274C Bus v1 (deprecated)",
    "MarasitUniversalBusNode": "\u274C Bus - UniversalBus v1 (deprecated)",
    "MarasitAnyBusNode": "\u274C AnyBus - UniversalBus v1 (deprecated)",
    "MaraScottAnyBusNode": "\ud83d\udc30 AnyBus - UniversalBus v2 (deprecated)/*",
    "MarasitDisplayInfoNode": "\u274C Display Info - Text v1 (deprecated)",
    "MarasitUpscalerRefinerNode": "\u274C Large Refiner - McBoaty v1 (deprecated)",
    "MarasitUpscalerRefinerNode_v2": "\u274C Large Refiner - McBoaty v2 (deprecated)",
}

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')
