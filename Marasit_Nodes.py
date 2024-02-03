#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# File: __init__.py
# Project: ComfyUI-MarasIT-Nodes
# By MarasIT (Discord: davask#4370)
# Copyright 2024 David Asquiedge (davask)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from .py.inc import web

web.init()

from .py.nodes.bus import Marasit_Bus

MANIFEST = {
    "name": "Maras IT Nodes",
    "version": (1,0,0),
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