#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from . import __SESSIONS_DIR__, __PROFILES_DIR__
from .py.nodes.UniversalBusNode import UniversalBusNode
from .py.inc.profiles.default import Node as default
from .py.inc.profiles.pipe_basic import Node as pipe_basic
from .py.inc.profiles.pipe_detailer import Node as pipe_Detailer
from .py.nodes.Bus_node import Bus_node
from .py.nodes.BusNode import BusNode
from .py.nodes.AnyBusNode import AnyBusNode
from .py.nodes.PipeNodeBasic import PipeNodeBasic
from .py.nodes.BusPipeNode import BusPipeNode

import os
import json
from aiohttp import web
import importlib
from server import PromptServer

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MarasitBusNode": Bus_node,
    "MarasitAnyBusNode": AnyBusNode,
    # "MarasitBusNode": BusNode,
    # "MarasitBusPipeNode": BusPipeNode,
    # "MarasitPipeNodeBasic": PipeNodeBasic,
    "MarasitUniversalBusNode": UniversalBusNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarasitBusNode": "Bus Node (DEPRECATED - use the AnyBus instead)",
    "MarasitAnyBusNode": "AnyBus Node - UniversalBus",
    # "MarasitBusNode": "Bus Node - Simple",
    # "MarasitBusPipeNode": "Bus/Pipe Node",
    # "MarasitPipeNodeBasic": "BasicPipe Node",
    "MarasitUniversalBusNode": "Universal Bus Node (DEPRECATED - use the AnyBus instead)"
}

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')
