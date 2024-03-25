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

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MarasitAnyBusNode": AnyBusNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarasitAnyBusNode": "AnyBus Node - UniversalBus",
}

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')
