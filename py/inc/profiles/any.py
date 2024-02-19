#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

from ..nodes import Configuration as _CONF

class Node:
    
    INPUT_TYPES = (
        _CONF.TYPES['ANY'], 
        _CONF.TYPES['ANY'], 
        _CONF.TYPES['ANY'],
        _CONF.TYPES['ANY'],
        _CONF.TYPES['ANY'],
        _CONF.TYPES['ANY'],
        _CONF.TYPES['ANY'],
        _CONF.TYPES['ANY'],
        _CONF.TYPES['ANY'],
        _CONF.TYPES['ANY'],
    )

    INPUT_NAMES = (
        "* (01)", 
        "* (02)", 
        "* (03)",
        "* (04)", 
        "* (05)", 
        "* (06)",
        "* (07)", 
        "* (08)", 
        "* (09)",
        "* (10)",
    )
    
    ENTRIES = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES)
    ENTRIES_JS = _CONF.generate_entries(INPUT_NAMES, INPUT_TYPES, 'js')    