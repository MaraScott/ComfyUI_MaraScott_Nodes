#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
# Bus.  Converts X connectors into one, and back again.  You can provide a bus as input
#       or the X separate inputs, or a combination.  If you provide a bus input and a separate
#       input (e.g. a model), the model will take precedence.
#
#       The term 'bus' comes from computer hardware, see https://en.wikipedia.org/wiki/pipe_(computing)
#       Largely inspired by Was Node Suite - Bus Node 
#
###

from ..inc.nodes import Configuration as _CONF
from ..inc.profiles.pipe_basic import Node as ProfileNodePipeBasic

class PipeNodeBasic:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {"id":"UNIQUE_ID"},
            "required":{},
            "optional": {
                "pipe (basic)" : ("BASIC_PIPE",),
                **ProfileNodePipeBasic.ENTRIES,
            }
        }

    RETURN_TYPES = ("BASIC_PIPE", ) + ProfileNodePipeBasic.INPUT_TYPES
    RETURN_NAMES = ("pipe (basic)",) + ProfileNodePipeBasic.INPUT_NAMES
    OUTPUT_NODE = _CONF.OUTPUT_NODE
    CATEGORY = _CONF.CATEGORY
    DESCRIPTION = "A Basic Pipe Node"
    FUNCTION = "pipe_fn"
    
    def pipe_fn(self, **kwargs):
        # Initialize the bus tuple with None values for each parameter
        bus = kwargs.get('pipe (basic)', (None,) * len(ProfileNodePipeBasic.INPUT_NAMES))
        if len(bus) != len(ProfileNodePipeBasic.INPUT_NAMES):
            raise ValueError("The 'pipe (basic)' tuple must have the same number of elements as '_INPUT_NAMES'")

        outputs = {}
        for name, pipe_value in zip(ProfileNodePipeBasic.INPUT_NAMES, bus):
            _input = kwargs.get(name, pipe_value)
            outputs[name] = _CONF.determine_output_value(name, _input, pipe_value)

        # _CONF.ensure_required_parameters(outputs)
        _CONF.handle_special_parameters(outputs)

        # Prepare and return the output bus tuple with updated values
        out_pipe = tuple(outputs[name] for name in ProfileNodePipeBasic.INPUT_NAMES)
        return (out_pipe,) + out_pipe
