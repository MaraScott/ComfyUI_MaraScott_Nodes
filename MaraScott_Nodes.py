#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from .py.nodes.Bus.AnyBus_v2 import AnyBus_v2
from .py.nodes.Info.DisplayInfo_v2 import DisplayInfo_v2
from .py.nodes.UpscalerRefiner.McBoaty_v3 import UpscalerRefiner_McBoaty_v3
from .py.nodes.UpscalerRefiner.McBoaty_v4 import McBoaty_Upscaler_v4, McBoaty_TilePrompter_v4, McBoaty_Refiner_v4
from .py.nodes.UpscalerRefiner.McBoaty_v5 import McBoaty_UpscalerRefiner_v5, McBoaty_Upscaler_v5, McBoaty_TilePrompter_v5, McBoaty_Refiner_v5
from .py.nodes.KSampler.InpaintingTileByMask_v1 import KSampler_setInpaintingTileByMask_v1, KSampler_pasteInpaintingTileByMask_v1
from .py.nodes.Prompt.PromptFromImage_v1 import PromptFromImage_v1
from .py.nodes.Prompt.TextConcatenate_v1 import TextConcatenate_v1
from .py.nodes.Prompt.TextConversion_v1 import TextConversion_StringToList_v1
from .py.nodes.Loop.ForLoop_v1 import ForLoopOpen_v1, ForLoopClose_v1, ForLoopWhileOpen_v1, ForLoopWhileClose_v1, ForLoopIntMathOperation_v1, ForLoopToBoolNode_v1
from .py.nodes.Util.Conditional import IsEmpty_v1, IsNone_v1, IsEmptyOrNone_v1, IsEqual_v1
from .py.vendor.ComfyUI_JNodes.blob.main.py.prompting_nodes import TokenCounter as TokenCounter_v1

from .py.nodes.Dev.DashBoard_v1 import DashBoard_v1

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {

    "MaraScottAnyBus_v2": AnyBus_v2,
    "MaraScottMcBoatyUpscalerRefiner_v5": McBoaty_UpscalerRefiner_v5,
    "MaraScottMcBoatyUpscaler_v5": McBoaty_Upscaler_v5,
    "MaraScottMcBoatyTilePrompter_v5": McBoaty_TilePrompter_v5,
    "MaraScottMcBoatyRefiner_v5": McBoaty_Refiner_v5,
    "MaraScottMcBoatyUpscaler_v4": McBoaty_Upscaler_v4,
    "MaraScottMcBoatyTilePrompter_v4": McBoaty_TilePrompter_v4,
    "MaraScottMcBoatyRefiner_v4": McBoaty_Refiner_v4,
    "MaraScottUpscalerRefinerNode_v3": UpscalerRefiner_McBoaty_v3,
    "MaraScottSetInpaintingByMask_v1": KSampler_setInpaintingTileByMask_v1,
    "MaraScottPasteInpaintingByMask_v1": KSampler_pasteInpaintingTileByMask_v1,

    "MaraScottForLoopOpen_v1": ForLoopOpen_v1,
    "MaraScottForLoopClose_v1": ForLoopClose_v1,
    "MaraScottForLoopWhileOpen_v1": ForLoopWhileOpen_v1,
    "MaraScottForLoopWhileClose_v1": ForLoopWhileClose_v1,
    "MaraScottForLoopIntMathOperation_v1": ForLoopIntMathOperation_v1,
    "MaraScottForLoopToBoolNode_v1": ForLoopToBoolNode_v1,
    
    "MaraScottIsEmpty_v1": IsEmpty_v1,
    "MaraScottIsNone_v1": IsNone_v1,
    "MaraScottIsEmptyOrNone_v1": IsEmptyOrNone_v1,
    "MaraScottIsEqual_v1": IsEqual_v1,

    "MaraScottPromptFromImage_v1": PromptFromImage_v1,
    "MaraScottTextConcatenate_v1": TextConcatenate_v1,
    "TextConversion_StringToList_v1": TextConversion_StringToList_v1,
    "MaraScottDisplayInfo_v2": DisplayInfo_v2,

    "MaraScottDashBoard_v1": DashBoard_v1,
    "MaraScott_Kijai_TokenCounter_v1": TokenCounter_v1,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes 
# active : \ud83d\udc30
# deprecated : \u274C
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaraScottDisplayInfo_v2": "\ud83d\udc30 Display Any - Text v2 /i",
    "MaraScottMcBoatyUpscalerRefiner_v5": "\ud83d\udc30 Large Refiner - McBoaty [1/3] v5 /u",
    "MaraScottMcBoatyUpscaler_v5": "\ud83d\udc30 Upscaler - McBoaty [1/3] v5 /u",
    "MaraScottMcBoatyTilePrompter_v5": "\ud83d\udc30 Tile Prompter - McBoaty [2/3] v5 /u",
    "MaraScottMcBoatyRefiner_v5": "\ud83d\udc30 Refiner - McBoaty [3/3] v5 /u",
    "MaraScottMcBoatyUpscaler_v4": "\ud83d\udc30 Upscaler - McBoaty [1/3] v4 /u",
    "MaraScottMcBoatyTilePrompter_v4": "\ud83d\udc30 Tile Prompter - McBoaty [2/3] v4 /u",
    "MaraScottMcBoatyRefiner_v4": "\ud83d\udc30 Refiner - McBoaty [3/3] v4 /u",
    "MaraScottMcBoatyUpscalerRefinerNode_v3": "\ud83d\udc30 Large Refiner - McBoaty v3 /u",
    "MaraScottSetInpaintingByMask_v1": "\ud83d\udc30 Set Inpainting Tile by mask - McInpainty [1/2] v1 /m",
    "MaraScottPasteInpaintingByMask_v1": "\ud83d\udc30 Paste Inpainting Tile by mask - McInpainty [2/2] v1 /m",

    "MaraScottForLoopOpen_v1": "\ud83d\udc30 For Loop Open v1 /l",
    "MaraScottForLoopClose_v1": "\ud83d\udc30 For Loop Close v1 /l",
    "MaraScottForLoopWhileOpen_v1": "\ud83d\udc30 For Loop While Open v1 /l",
    "MaraScottForLoopWhileClose_v1": "\ud83d\udc30 For Loop While Close v1 /l",
    "MaraScottForLoopIntMathOperation_v1": "\ud83d\udc30 For Loop IntMathOperation v1 /l",
    "MaraScottForLoopToBoolNode_v1": "\ud83d\udc30 For Loop ToBoolNode v1 /l",

    "MaraScottIsEmpty_v1": "\ud83d\udc30 Is Empty v1 /c",
    "MaraScottIsNone_v1": "\ud83d\udc30 Is None v1 /c",
    "MaraScottIsEmptyOrNone_v1": "\ud83d\udc30 Is Empty Or None v1 /c",
    "MaraScottIsEqual_v1": "\ud83d\udc30 Is Equal v1 /c",

    "MaraScottPromptFromImage_v1": "\ud83d\udc30 Prompt From Image - McPrompty v1 /p",
    "MaraScottTextConcatenate_v1": "\ud83d\udc30 Text Concatenation v1 /t",
    "TextConversion_StringToList_v1": "\ud83d\udc30 Multiline to List v1 /t",
    "MaraScottAnyBus_v2": "\ud83d\udc30 AnyBus - UniversalBus v2 /*",

    "MaraScottDashBoard_v1": "\ud83d\udc30 DashBoard v1 /d",
    "MaraScott_Kijai_TokenCounter_v1": "\ud83d\udc30 TokenCounter (from kijai/ComfyUI-KJNodes) /v",

}

print('\033[34m[MaraScott] \033[92mLoaded\033[0m')
