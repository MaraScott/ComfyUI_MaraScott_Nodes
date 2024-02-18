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
        "CONDITIONING", 
        "CONDITIONING", 
    )

    INPUT_NAMES = (
        "model", 
        "clip", 
        "vae", 
        "positive", 
        "negative", 
    )
    
    ENTRIES = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES)
    ENTRIES_JS = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES, 'js')    