#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# File: __init__.py
# Project: ComfyUI-MarasIT-Nodes
# Author: David Asquiedge
# Copyright (c) 2023 Mel Massadian
#
###

import os 
import sys 

python = sys.executable
p310_plus = (sys.version_info >= (3, 10))

__ROOT__file__ = __file__

from .Marasit_Nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
