#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Bus.  Converts X connectors into one, and back again.  You can provide a bus as input
#       or the X separate inputs, or a combination.  If you provide a bus input and a separate
#       input (e.g. a model), the model will take precedence.
#
#       The term 'bus' comes from computer hardware, see https://en.wikipedia.org/wiki/Bus_(computing)
#       Largely inspired by Was Node Suite - Bus Node 
#
###

from ..inc.nodes import Configuration as _CONF
from ..inc.profiles.default import Node as ProfileNodeDefault
from ..inc.profiles.pipe_basic import Node as ProfileNodePipeBasic

class BusPipeNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {"id":"UNIQUE_ID"},
            "required":{},
            "optional": {
                "bus": ("BUS",),
                "pipe (basic)": ("BASIC_PIPE",),
                **ProfileNodeDefault.ENTRIES,
            }
        }

    RETURN_TYPES = ("BUS", ) + ("BASIC_PIPE", ) + ProfileNodeDefault.INPUT_TYPES
    RETURN_NAMES = ("bus",) + ("pipe (basic)",) + ProfileNodeDefault.INPUT_NAMES
    OUTPUT_NODE = _CONF.OUTPUT_NODE
    CATEGORY = _CONF.CATEGORY
    DESCRIPTION = "A Bus/Pipe Node"
    FUNCTION = "buspipe_fn"
    
    def buspipe_fn(self, **kwargs):
        # Initialize the bus tuple with None values for each parameter
        bus = kwargs.get('bus', (None,) * len(ProfileNodeDefault.INPUT_NAMES))
        if len(bus) != len(ProfileNodeDefault.INPUT_NAMES):
            raise ValueError("The 'bus' tuple must have the same number of elements as '_INPUT_NAMES'")

        bus_outputs = {}
        pipe_basic_outputs = {}
        for name, bus_value in zip(ProfileNodeDefault.INPUT_NAMES, bus):
            _input = kwargs.get(name, bus_value)
            bus_outputs[name] = _CONF.determine_output_value(name, _input, bus_value)
            if name in ['model', 'clip', 'vae', 'positive', 'negative']:
                pipe_basic_outputs[name] = bus_outputs[name]

        # _CONF.ensure_required_parameters(bus_outputs)
        _CONF.handle_special_parameters(bus_outputs)

        # Prepare and return the output bus tuple with updated values
        out_bus = tuple(bus_outputs[name] for name in ProfileNodeDefault.INPUT_NAMES)
        out_pipe_basic = tuple(pipe_basic_outputs[name] for name in ProfileNodePipeBasic.INPUT_NAMES)
        return (out_bus,) + (out_pipe_basic,) + out_bus
