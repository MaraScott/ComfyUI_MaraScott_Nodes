#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

from ..nodes import Configuration as _CONF
from .default import Node as default
from .pipe_basic import Node as pipe_basic
from .pipe_detailer import Node as pipe_detailer

class Node:
    
    _ENTRIES = {**default.ENTRIES_JS, **pipe_basic.ENTRIES_JS, **pipe_detailer.ENTRIES_JS}

    # Convert to sets to remove duplicates and merge
    INPUT_TYPES = tuple(list(_ENTRIES.values()))

    INPUT_NAMES = tuple(list(_ENTRIES.keys()))
        
    ENTRIES = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES)
    ENTRIES_JS = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES, 'js')