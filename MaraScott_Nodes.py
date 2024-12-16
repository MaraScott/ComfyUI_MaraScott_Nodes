#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from .py.utils.constants import NAMESPACE, get_name, get_category

from .py.nodes.Image.LoadImage_v1 import LoadImage_v1
from .py.nodes.Bus.AnyBus_v2 import AnyBus_v2
from .py.nodes.Info.DisplayInfo_v2 import DisplayInfo_v2
from .py.nodes.UpscalerRefiner.McBoaty_v3 import UpscalerRefiner_McBoaty_v3
from .py.nodes.UpscalerRefiner.McBoaty_v5 import McBoaty_UpscalerRefiner_v5, McBoaty_Upscaler_v5, McBoaty_TilePrompter_v5, McBoaty_Refiner_v5
from .py.nodes.UpscalerRefiner.McBoaty_v6 import Mara_Tiler_v1, Mara_Untiler_v1, Mara_McBoaty_v6, Mara_McBoaty_Configurator_v6, Mara_McBoaty_TilePrompter_v6, Mara_McBoaty_Refiner_v6
from .py.nodes.Inpainting.InpaintingTileByMask_v1 import KSampler_setInpaintingTileByMask_v1, KSampler_pasteInpaintingTileByMask_v1
from .py.nodes.Prompt.PromptFromImage_v1 import PromptFromImage_v1
from .py.nodes.Prompt.TextConcatenate_v1 import TextConcatenate_v1
from .py.nodes.Prompt.TextConversion_v1 import TextConversion_StringToList_v1
from .py.nodes.Loop.ForLoop_v1 import ForLoopOpen_v1, ForLoopClose_v1, ForLoopWhileOpen_v1, ForLoopWhileClose_v1, ForLoopIntMathOperation_v1, ForLoopToBoolNode_v1
from .py.nodes.Util.Conditional import IsEmpty_v1, IsNone_v1, IsEmptyOrNone_v1, IsEqual_v1
from .py.nodes.Util.Image import ImageToGradient_v1
from .py.nodes.Util.Model import GetModelBlocks_v1

from .py.vendor.ComfyUI_JNodes.blob.main.py.prompting_nodes import TokenCounter as TokenCounter_v1
from .py.vendor.ComfyUI_Florence2.nodes import DownloadAndLoadFlorence2Model as DownloadAndLoadFlorence2Model_v1, Florence2Run as Florence2Run_v1
from .py.vendor.kohya_hiresfix.kohya_hiresfix import Hires as Hires_v1
# from .py.vendor.comfyui_ollama.CompfyuiOllama import Mara_OllamaVision_v1

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {

    f"{NAMESPACE}AnyBus_v2": AnyBus_v2,
    f"{NAMESPACE}Tiler_v1": Mara_Tiler_v1,
    f"{NAMESPACE}Untiler_v1": Mara_Untiler_v1,
    f"{NAMESPACE}McBoaty_v6": Mara_McBoaty_v6,
    f"{NAMESPACE}McBoatyConfigurator_v6": Mara_McBoaty_Configurator_v6,
    f"{NAMESPACE}McBoatyTilePrompter_v6": Mara_McBoaty_TilePrompter_v6,
    f"{NAMESPACE}McBoatyRefiner_v6": Mara_McBoaty_Refiner_v6,
    f"{NAMESPACE}McBoatyUpscalerRefiner_v5": McBoaty_UpscalerRefiner_v5,
    f"{NAMESPACE}McBoatyUpscaler_v5": McBoaty_Upscaler_v5,
    f"{NAMESPACE}McBoatyTilePrompter_v5": McBoaty_TilePrompter_v5,
    f"{NAMESPACE}McBoatyRefiner_v5": McBoaty_Refiner_v5,
    f"{NAMESPACE}McBoatyUpscalerRefinerNode_v3": UpscalerRefiner_McBoaty_v3,
    f"{NAMESPACE}SetInpaintingByMask_v1": KSampler_setInpaintingTileByMask_v1,
    f"{NAMESPACE}PasteInpaintingByMask_v1": KSampler_pasteInpaintingTileByMask_v1,

    f"{NAMESPACE}ForLoopOpen_v1": ForLoopOpen_v1,
    f"{NAMESPACE}ForLoopClose_v1": ForLoopClose_v1,
    f"{NAMESPACE}ForLoopWhileOpen_v1": ForLoopWhileOpen_v1,
    f"{NAMESPACE}ForLoopWhileClose_v1": ForLoopWhileClose_v1,
    f"{NAMESPACE}ForLoopIntMathOperation_v1": ForLoopIntMathOperation_v1,
    f"{NAMESPACE}ForLoopToBoolNode_v1": ForLoopToBoolNode_v1,
    
    f"{NAMESPACE}IsEmpty_v1": IsEmpty_v1,
    f"{NAMESPACE}IsNone_v1": IsNone_v1,
    f"{NAMESPACE}IsEmptyOrNone_v1": IsEmptyOrNone_v1,
    f"{NAMESPACE}IsEqual_v1": IsEqual_v1,

    f"{NAMESPACE}PromptFromImage_v1": PromptFromImage_v1,
    f"{NAMESPACE}TextConcatenate_v1": TextConcatenate_v1,
    f"{NAMESPACE}TextConversion_StringToList_v1": TextConversion_StringToList_v1,
    f"{NAMESPACE}ImageToGradient_v1": ImageToGradient_v1,
    f"{NAMESPACE}DisplayInfo_v2": DisplayInfo_v2,
    
    f"{NAMESPACE}GetModelBlocks_v1": GetModelBlocks_v1,

    f"{NAMESPACE}LoadImage_v1": LoadImage_v1,
}

VENDOR_NODE_CLASS_MAPPINGS = {
    f"{NAMESPACE}_Kijai_TokenCounter_v1": TokenCounter_v1,
    f"{NAMESPACE}_Kijai_DownloadAndLoadFlorence2Model_v1": DownloadAndLoadFlorence2Model_v1,
    f"{NAMESPACE}_Kijai_Florence2Run_v1": Florence2Run_v1,    
    f"{NAMESPACE}_laksjdjf_Hires_v1": Hires_v1,
    # f"{NAMESPACE}_stavsap_OllamaVision_v1": Mara_OllamaVision_v1,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes 
# active : \ud83d\udc30
# deprecated : \u274C

# Define getter and setter functions
def _get_CATEGORY(cls):
    return cls._CATEGORY

def _set_CATEGORY(cls, value):
    cls._CATEGORY = value
    
for key, value in VENDOR_NODE_CLASS_MAPPINGS.items():
    setattr(value, 'CATEGORY', property(classmethod(_get_CATEGORY), classmethod(_set_CATEGORY)))
    value.CATEGORY = get_category("vendor")

NODE_DISPLAY_NAME_MAPPINGS = {
    key: get_name(value, getattr(value, "NAME", value.__name__), getattr(value, "SHORTCUT", "")) for key, value in NODE_CLASS_MAPPINGS.items()
}

VENDOR_NODE_DISPLAY_NAME_MAPPINGS = {
    "MaraScott_Kijai_TokenCounter_v1": get_name(TokenCounter_v1, "TokenCounter", "v", "kijai"),
    "MaraScott_Kijai_DownloadAndLoadFlorence2Model_v1": get_name(DownloadAndLoadFlorence2Model_v1, "DownloadAndLoadFlorence2Model", "v", "Kijai"),
    "MaraScott_Kijai_Florence2Run_v1": get_name(Florence2Run_v1, "Florence2Run", "v", "Kijai"),
    "MaraScott_laksjdjf_Hires_v1": get_name(Hires_v1, "Apply Kohya's HiresFix - sd1.5 only", "v", "laksjdjf"),
}

NODE_CLASS_MAPPINGS.update(VENDOR_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(VENDOR_NODE_DISPLAY_NAME_MAPPINGS)

print('\033[34m[MaraScott] \033[92mLoaded\033[0m')
