#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

from ..nodes import Configuration as _CONF

class Node:
    
    INPUT_ANY_QTY = 12

    INPUT_NAMES = tuple("* {:02d}".format(i) for i in range(1, INPUT_ANY_QTY + 1))
    
    INPUT_TYPES = (_CONF.TYPES['ANY'],) * len(INPUT_NAMES)

    ENTRIES = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES)
    ENTRIES_JS = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES, 'js')    