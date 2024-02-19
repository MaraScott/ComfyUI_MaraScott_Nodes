#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

from ..nodes import Configuration as _CONF

class Node:
        
    INPUT_TYPES = (
        "MODEL", 
        "CLIP", 
        "VAE", 
        "LATENT", 
        "IMAGE", 
        "MASK", 
        _CONF.TYPES['ANY'], 
        _CONF.TYPES['ANY'], 
        _CONF.TYPES['ANY'],
        "CONDITIONING", 
        "CONDITIONING", 
        "STRING", 
        "STRING", 
        "INT",
        "INT",
    )

    INPUT_NAMES = (
        "model", 
        "clip", 
        "vae", 
        "latent",
        "image", 
        "mask", 
        "* (1)", 
        "* (2)", 
        "* (3)",
        "positive", 
        "negative", 
        "text (positive)", 
        "text (negative)", 
        "width", 
        "height", 
    )
    
    ENTRIES = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES)
    ENTRIES_JS = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES, 'js')