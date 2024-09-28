#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Bus. Converts X connectors into one, and back again. You can provide a bus as input
# or the X separate inputs, or a combination. If you provide a bus input and a separate
# input (e.g., a model), the model will take precedence.
#
# The term 'bus' comes from computer hardware, see https://en.wikipedia.org/wiki/Bus_(computing)
# Largely inspired by Was Node Suite - Bus Node 
#
###

import os
import json
from ...inc.nodes import Configuration as _CONF
from ...utils.Api import Api
from ...utils.path import PROFILES_DIR
from ...inc.profiles.any import Node as ProfileNodeAny
from ...utils.log import *

Api()
supported_profile_extensions = {'.json'}

def get_profile_list_():
    profiles = {}
    for root, _, files in os.walk(PROFILES_DIR):
        for file in files:
            if any(file.endswith(ext) for ext in supported_profile_extensions):
                profile_name = os.path.splitext(file)[0]
                profiles[profile_name] = os.path.join(root, file)
    return profiles

def get_profile_list():
    profiles = get_profile_list_()
    return list(profiles.keys())

class AnyBus_v2:

    PROFILES_FILE = get_profile_list()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {"id": "UNIQUE_ID"},
            "required": {
                "profiler": (cls.PROFILES_FILE, )
            },
            "optional": {
                "bus": ("BUS",),
                **ProfileNodeAny.ENTRIES,
            }
        }

    RETURN_TYPES = (
        ("BUS", )
        + ProfileNodeAny.INPUT_TYPES
    )

    RETURN_NAMES = (
        ("bus", )
        + ProfileNodeAny.INPUT_NAMES
    )

    OUTPUT_NODE = _CONF.OUTPUT_NODE
    CATEGORY = _CONF.CATEGORY
    DESCRIPTION = 'An "ANY" Bus Node'
    FUNCTION = "fn"

    profiles = {}

    def __init__(self):
        self.load_profiles()

    @classmethod
    def load_profiles(cls, name="default"):
        cls.profiles = {}
        if name in cls.PROFILES_FILE:
            profile_names = list(cls.PROFILES_FILE)
            index = profile_names.index(name)
            with open(os.path.join(PROFILES_DIR, f"{cls.PROFILES_FILE[index]}.json"), 'r') as file:
                cls.profiles[name] = json.load(file)

    @classmethod
    def save_profiles(cls, name):
        if name in cls.PROFILES_FILE:
            profile_names = list(cls.PROFILES_FILE)
            index = profile_names.index(name)
            with open(os.path.join(PROFILES_DIR, f"{cls.PROFILES_FILE[index]}.json"), 'w') as file:
                json.dump(cls.profiles[name], file, indent=4)

    @classmethod
    def add_profile(cls, name, settings):
        cls.profiles[name] = settings
        cls.save_profiles(name)

    @classmethod
    def get_profile(cls, name):
        return cls.profiles.get(name, None)

    def fn(self, **kwargs):
        # Initialize the bus tuple with None values for each parameter
        profile_name = kwargs.get('profiler', 'default')
        if profile_name != 'default':
            profile_settings = self.get_profile(profile_name)
            if profile_settings:
                kwargs.update(profile_settings)

        bus = kwargs.get('bus', (None, ) * len(ProfileNodeAny.INPUT_NAMES))
        if len(bus) != len(ProfileNodeAny.INPUT_NAMES):
            raise ValueError("The 'bus' tuple must have the same number of elements as 'INPUT_NAMES'")

        outputs = {}
        for name, bus_value in zip(ProfileNodeAny.INPUT_NAMES, bus):
            _input = _CONF.get_kwarg_with_prefix(kwargs, name, bus_value)
            outputs[name] = _CONF.determine_output_value(name, _input, bus_value)

        _CONF.handle_special_parameters(outputs)

        # Prepare and return the output bus tuple with updated values
        out_bus = tuple(outputs[name] for name in ProfileNodeAny.INPUT_NAMES)

        return (
            (out_bus, )
            + out_bus
        )
