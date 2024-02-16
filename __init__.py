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
###
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
#
###

import os 
import sys 

python = sys.executable
p310_plus = (sys.version_info >= (3, 10))

__ROOT__file__ = __file__

# Directory where you want to save the file
base_dir = os.path.abspath(os.path.dirname(__ROOT__file__))
web_dir = os.path.join(base_dir, "..", "..", "web", "extensions", "marasit")
web_dir = os.path.realpath(web_dir)
if not os.path.exists(web_dir):
    os.makedirs(web_dir)
__WEB_DIR__ = web_dir

sessions_dir = os.path.join(web_dir, "sessions")
if not os.path.exists(sessions_dir):
    os.makedirs(sessions_dir)
__SESSIONS_DIR__ = sessions_dir

profiles_dir = os.path.join(web_dir, "profiles")
if not os.path.exists(profiles_dir):
    os.makedirs(profiles_dir)
__PROFILES_DIR__ = profiles_dir

from .Marasit_Nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

MANIFEST = {
    "name": "MarasIT Nodes",
    "version": (1,3,0),
    "author": "davask",
    "project": "https://github.com/davask/ComfyUI-MarasIT-Nodes",
    "description": "A simple Bus node",
}
