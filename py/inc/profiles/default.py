#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

from ..nodes import Configuration as _CONF

class Node:
    
    _TYPES = _CONF.TYPES
    
    INPUT_TYPES = (
        "MODEL", 
        "CLIP", 
        "VAE", 
        "CONDITIONING", 
        "CONDITIONING", 
        "STRING", 
        "STRING", 
        "LATENT", 
        "IMAGE", 
        "MASK", 
        _TYPES['ANY'], 
        _TYPES['ANY'], 
        _TYPES['ANY'],
    )

    INPUT_NAMES = (
        "model", 
        "clip", 
        "vae", 
        "positive", 
        "negative", 
        "text (positive)", 
        "text (negative)", 
        "latent", 
        "image", 
        "mask", 
        "* (1)", 
        "* (2)", 
        "* (3)"
    )
    
    ENTRIES = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES)
    ENTRIES_JS = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES, 'js')