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
from ..inc.profiles.any import Node as ProfileNodeAny
# from ..inc.profiles.pipe_basic import Node as ProfileNodePipeBasic
# from ..inc.profiles.pipe_detailer import Node as ProfileNodePipeDetailer

class AnyBusNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {"id":"UNIQUE_ID"},
            "required":{},
            "optional": {
                "bus": ("BUS",),
                # "pipe (basic)": "BASIC_PIPE",
                # "pipe (detailer)": "DETAILER_PIPE",
                **ProfileNodeAny.ENTRIES,
            }
        }

    RETURN_TYPES = (
        ("BUS", ) 
        # + ("BASIC_PIPE", ) 
        # + ("DETAILER_PIPE", ) 
        + ProfileNodeAny.INPUT_TYPES
    )
    
    RETURN_NAMES = (
        ("bus",) 
        # + ("pipe (basic)", )
        # + ("pipe (detailer)", )
        + ProfileNodeAny.INPUT_NAMES
    )
    
    OUTPUT_NODE = _CONF.OUTPUT_NODE
    CATEGORY = _CONF.CATEGORY
    DESCRIPTION = "An \"ANY\" Bus Node"
    FUNCTION = "bus_fn"
    
    def bus_fn(self, **kwargs):
        # Initialize the bus tuple with None values for each parameter
        bus = kwargs.get('bus', (None,) * len(ProfileNodeAny.INPUT_NAMES))
        if len(bus) != len(ProfileNodeAny.INPUT_NAMES):
            raise ValueError("The 'bus' tuple must have the same number of elements as '_INPUT_NAMES'")

        outputs = {}
        # pipe_basic_outputs = {}
        # pipe_detailer_outputs = {}
        for name, bus_value in zip(ProfileNodeAny.INPUT_NAMES, bus):
            _input = _CONF.get_kwarg_with_prefix(kwargs, name, bus_value)
            outputs[name] = _CONF.determine_output_value(name, _input, bus_value)
            # if name in ProfileNodePipeBasic.INPUT_NAMES:
            #     pipe_basic_outputs[name] = outputs[name]

        # _CONF.ensure_required_parameters(outputs)
        _CONF.handle_special_parameters(outputs)

        # Prepare and return the output bus tuple with updated values
        out_bus = tuple(outputs[name] for name in ProfileNodeAny.INPUT_NAMES)
        # out_pipe_basic = tuple(pipe_basic_outputs[name] for name in ProfileNodePipeBasic.INPUT_NAMES)
        # out_pipe_detailer = tuple(pipe_detailer_outputs[name] for name in ProfileNodePipeDetailer.INPUT_NAMES)
        
        return (
            (out_bus,) 
            # + (out_pipe_basic,) 
            # + (out_pipe_detailer,) 
            + out_bus
        )
