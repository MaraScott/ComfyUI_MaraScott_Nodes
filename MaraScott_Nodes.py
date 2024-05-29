#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from . import __SESSIONS_DIR__, __PROFILES_DIR__
from .py.nodes.Bus.AnyBus_v2 import AnyBus_v2
from .py.nodes.Info.DisplayInfoNode_v1 import DisplayInfoNode_v1
from .py.nodes.UpscalerRefiner.McBoaty_v1 import UpscalerRefiner_McBoaty_v1
from .py.nodes.UpscalerRefiner.McBoaty_v2 import UpscalerRefiner_McBoaty_v2
from .py.nodes.KSampler.InpaintingTileByMask_v1 import KSampler_setInpaintingTileByMask_v1, KSampler_pasteInpaintingTileByMask_v1

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MaraScottAnyBus_v2": AnyBus_v2,
    "MaraScottDisplayInfo_v1": DisplayInfoNode_v1,
    "MaraScottUpscalerRefinerNode_v2": UpscalerRefiner_McBoaty_v2,
    "MaraScottSetInpaintingByMask_v1": KSampler_setInpaintingTileByMask_v1,
    "MaraScottPasteInpaintingByMask_v1": KSampler_pasteInpaintingTileByMask_v1,

    "MaraScottAnyBusNode": AnyBus_v2,
    "MaraScottDisplayInfoNode": DisplayInfoNode_v1,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes 
# active : \ud83d\udc30
# deprecated : \u274C
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaraScottAnyBus_v2": "\ud83d\udc30 AnyBus - UniversalBus v2 /*",
    "MaraScottDisplayInfo_v1": "\ud83d\udc30 Display Info - Text v1 /i",
    "MaraScottUpscalerRefinerNode_v2": "\ud83d\udc30 Large Refiner - McBoaty v2 /u",
    "MaraScottSetInpaintingByMask_v1": "\ud83d\udc30 Set Inpainting Tile by mask [1/2] v1 /m",
    "MaraScottPasteInpaintingByMask_v1": "\ud83d\udc30 Paste Inpainting Tile by mask [2/2] v1 /m",

    "MaraScottAnyBusNode": "\u274C AnyBus - UniversalBus v2 /*",
    "MaraScottDisplayInfoNode": "\u274C Display Info - Text v1 /i",
}

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')
