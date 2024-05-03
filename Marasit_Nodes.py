#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from . import __SESSIONS_DIR__, __PROFILES_DIR__
from .py.nodes.AnyBusNode import AnyBusNode
from .py.nodes.DisplayInfoNode import DisplayInfoNode
from .py.nodes.UpscalerRefiner.McBoaty_v1 import UpscalerRefiner_McBoaty_v1
from .py.nodes.UpscalerRefiner.McBoaty_v2 import UpscalerRefiner_McBoaty_v2

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MarasitAnyBusNode": AnyBusNode,
    "MarasitDisplayInfoNode": DisplayInfoNode,
    "MarasitUpscalerRefinerNode": UpscalerRefiner_McBoaty_v1,
    "MarasitUpscalerRefinerNode_v2": UpscalerRefiner_McBoaty_v2
}

# A dictionary that contains the friendly/humanly readable titles for the nodes 
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarasitAnyBusNode": "\ud83d\udc30 AnyBus - UniversalBus /*",
    "MarasitDisplayInfoNode": "\ud83d\udc30 Display Info - Text /i",
    "MarasitUpscalerRefinerNode": "\ud83d\udc30 Large Refiner - McBoaty_v1 (deprecated)/u",
    "MarasitUpscalerRefinerNode_v2": "\ud83d\udc30 Large Refiner - McBoaty_v2 /u",
}

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')
