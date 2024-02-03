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

from .py.nodes.bus import Bus_node

MANIFEST = {
    "name": "Maras IT Nodes",
    "version": (1,1,1),
    "author": "davask",
    "project": "https://github.com/davask/ComfyUI-MarasIT-Nodes",
    "description": "A simple Bus node",
}

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MarasitBusNode": Bus_node,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarasitBusNode": "Bus Node"
}

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')