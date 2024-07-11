#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from .py.nodes.Bus.AnyBus_v2 import AnyBus_v2
from .py.nodes.Info.DisplayInfoNode_v1 import DisplayInfoNode_v1
from .py.nodes.UpscalerRefiner.McBoaty_v2 import UpscalerRefiner_McBoaty_v2
from .py.nodes.UpscalerRefiner.McBoaty_v3 import UpscalerRefiner_McBoaty_v3
from .py.nodes.UpscalerRefiner.McBoaty_v4 import McBoaty_Upscaler_v4, McBoaty_TilePrompter_v4, McBoaty_Refiner_v4
from .py.nodes.UpscalerRefiner.McBoaty_v5 import UpscalerRefiner_McBoaty_v5, McBoaty_Upscaler_v5, McBoaty_TilePrompter_v5, McBoaty_Refiner_v5
from .py.nodes.KSampler.InpaintingTileByMask_v1 import KSampler_setInpaintingTileByMask_v1, KSampler_pasteInpaintingTileByMask_v1
from .py.nodes.Prompt.PromptFromImage_v1 import PromptFromImage_v1
from .py.vendor.ComfyUI_JNodes.blob.main.py.prompting_nodes import TokenCounter as TokenCounter_v1

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MaraScottAnyBus_v2": AnyBus_v2,
    "MaraScottDisplayInfo_v1": DisplayInfoNode_v1,
    "UpscalerRefiner_McBoaty_v5": UpscalerRefiner_McBoaty_v5,
    "McBoaty_Upscaler_v5": McBoaty_Upscaler_v5,
    "McBoaty_TilePrompter_v5": McBoaty_TilePrompter_v5,
    "McBoaty_Refiner_v5": McBoaty_Refiner_v5,
    "McBoaty_Upscaler_v4": McBoaty_Upscaler_v4,
    "McBoaty_TilePrompter_v4": McBoaty_TilePrompter_v4,
    "McBoaty_Refiner_v4": McBoaty_Refiner_v4,
    "MaraScottUpscalerRefinerNode_v3": UpscalerRefiner_McBoaty_v3,
    "MaraScottSetInpaintingByMask_v1": KSampler_setInpaintingTileByMask_v1,
    "MaraScottPasteInpaintingByMask_v1": KSampler_pasteInpaintingTileByMask_v1,
    "MaraScottPromptFromImage_v1": PromptFromImage_v1,

    "MaraScott_Kijai_TokenCounter_v1": TokenCounter_v1,

    "MaraScottAnyBusNode": AnyBus_v2,
    "MaraScottDisplayInfoNode": DisplayInfoNode_v1,
    "MaraScottUpscalerRefinerNode_v2": UpscalerRefiner_McBoaty_v2,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes 
# active : \ud83d\udc30
# deprecated : \u274C
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaraScottAnyBus_v2": "\ud83d\udc30 AnyBus - UniversalBus v2 /*",
    "MaraScottDisplayInfo_v1": "\ud83d\udc30 Display Info - Text v1 /i",
    "MaraScottUpscalerRefinerNode_v3": "\ud83d\udc30 Large Refiner - McBoaty v3 /u",
    "McBoaty_Upscaler_v5": "\ud83d\udc30 Upscaler - McBoaty [1/3] v5 /u",
    "McBoaty_TilePrompter_v5": "\ud83d\udc30 Tile Prompter - McBoaty [2/3] v5 /u",
    "McBoaty_Refiner_v5": "\ud83d\udc30 Refiner - McBoaty [3/3] v5 /u",
    "McBoaty_Upscaler_v4": "\ud83d\udc30 Upscaler - McBoaty [1/3] v4 /u",
    "McBoaty_TilePrompter_v4": "\ud83d\udc30 Tile Prompter - McBoaty [2/3] v4 /u",
    "McBoaty_Refiner_v4": "\ud83d\udc30 Refiner - McBoaty [3/3] v4 /u",
    "UpscalerRefiner_McBoaty_v5": "\ud83d\udc30 Large Refiner - McBoaty [1/3] v5 /u",
    "MaraScottSetInpaintingByMask_v1": "\ud83d\udc30 Set Inpainting Tile by mask - McInpainty [1/2] v1 /m",
    "MaraScottPasteInpaintingByMask_v1": "\ud83d\udc30 Paste Inpainting Tile by mask - McInpainty [2/2] v1 /m",
    "MaraScottPromptFromImage_v1": "\ud83d\udc30 Prompt From Image - McPrompty v1 /p",

    "MaraScott_Kijai_TokenCounter_v1": "\ud83d\udc30 TokenCounter (from kijai/ComfyUI-KJNodes) /v",

    "MaraScottAnyBusNode": "\u274C AnyBus - UniversalBus v2 /*",
    "MaraScottDisplayInfoNode": "\u274C Display Info - Text v1 /i",
    "MaraScottUpscalerRefinerNode_v2": "\u274C Large Refiner - McBoaty v2 /u",
}

print('\033[34m[MaraScott] \033[92mLoaded\033[0m')
