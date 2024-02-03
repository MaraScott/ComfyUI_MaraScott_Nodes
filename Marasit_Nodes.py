#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from .py.inc import web

web.init()

from .py.nodes.bus import Marasit_Bus

MANIFEST = {
    "name": "Maras IT Nodes",
    "version": (1,1,0),
    "author": "davask",
    "project": "https://github.com/davask/ComfyUI-MarasIT-Nodes",
    "description": "A simple Bus node",
}

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MarasitBusNode": Marasit_Bus,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarasitBusNode": "Bus Node (By Maras IT)"
}

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')