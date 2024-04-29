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
from .py.nodes.UpscalerRefinerNode import UpscalerRefinerNode

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MarasitAnyBusNode": AnyBusNode,
    "MarasitDisplayInfoNode": DisplayInfoNode,
    "MarasitUpscalerRefinerNode": UpscalerRefinerNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes 
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarasitAnyBusNode": "\ud83d\udc30 AnyBus Node - UniversalBus /*",
    "MarasitDisplayInfoNode": "\ud83d\udc30 Display Info - Text /i",
    "MarasitUpscalerRefinerNode": "\ud83d\udc30 UpScaler Refiner - McBoaty /u"
}

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')
