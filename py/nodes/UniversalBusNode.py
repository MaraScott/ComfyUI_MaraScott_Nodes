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

from ... import __SESSIONS_DIR__, __PROFILES_DIR__

from ..inc.nodes import Configuration as _CONF
from ..inc.profiles.universal import Node as ProfileNodeUniversal
from ..inc.profiles.pipe_basic import Node as ProfileNodePipeBasic

import os
import json
import torch

class UniversalBusNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "hidden": {"id":"UNIQUE_ID"},
            "required":{},
            "optional": {
                "bus": ("BUS",),
                "pipe (basic)": ("BASIC_PIPE",),
                **ProfileNodeUniversal.ENTRIES,
            }
        }

    RETURN_TYPES = ("BUS", ) + ("BASIC_PIPE", ) + ProfileNodeUniversal.INPUT_TYPES
    RETURN_NAMES = ("bus",) + ("pipe (basic)", ) + ProfileNodeUniversal.INPUT_NAMES
    OUTPUT_NODE = _CONF.OUTPUT_NODE
    CATEGORY = _CONF.CATEGORY
    FUNCTION = "universal_bus_fn"
    DESCRIPTION = "A Universal Bus/Pipe Node"
    
    IS_UNIVERSAL = False
    
    def universal_bus_fn(self, **kwargs):
        
        if self.IS_UNIVERSAL:
            session_id = "unique"  # As mentioned, session_id is always "unique"
            node_id = kwargs.get('id', None)
            profile = 'default'

            # Constructing the file path for session_unique_node_{nid}.json
            filename = f"session_{session_id}_node_{node_id}.json"
            filepath = os.path.join(__SESSIONS_DIR__, filename)

            # Load inputs from the file
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    profileByNode = json.load(file)
            else:
                raise FileNotFoundError(f"The file {filepath} does not exist.")

            # Constructing the file path for session_unique_node_{nid}.json
            filename = f"profile_{profileByNode}.json"
            filepath = os.path.join(__PROFILES_DIR__, filename)

            # Load inputs from the file
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    inputsByProfile = json.load(file)
        else:
            inputsByProfile = {
                "bus": "BUS",
                "pipe (basic)": "BASIC_PIPE",
                **ProfileNodeUniversal.ENTRIES,
            }

        # Initialize the bus tuple with None values for each parameter
        inputs = {}
        for name in inputsByProfile.keys():
            if name != 'bus' and not name.startswith('pipe'):
                inputs[name] = None
        bus_outputs = inputs.copy()
        pipe_basic_outputs = {}
        
        in_bus = kwargs.get('bus', (None,) * len(inputs))

        # print(in_bus)
        # Update bus_outputs based on in_bus values
        for i, name in enumerate(inputs):
            if in_bus[i] is not None:  # Only update if in_bus value is not None
                bus_outputs[name] = in_bus[i]
                if name in ['model', 'clip', 'vae', 'positive', 'negative']:
                    pipe_basic_outputs[name] = bus_outputs[name]
                
        in_pipe = kwargs.get('pipe (basic)', (None,) * len(inputs))
        # print(in_pipe)
        # Update outputs based on in_bus values
        for i, name in enumerate(inputs):
            if in_pipe[i] is not None:  # Only update if in_bus value is not None
                bus_outputs[name] = in_pipe[i]
                if name in ['model', 'clip', 'vae', 'positive', 'negative']:
                    pipe_basic_outputs[name] = bus_outputs[name]

        # Update bus_outputs based on inputs and current bus_outputs
        for name, value in inputs.items():
            inputs[name] = kwargs.get(name, None)
            bus_outputs[name] = _CONF.determine_output_value(name, inputs[name], bus_outputs[name])
            if name in ['model', 'clip', 'vae', 'positive', 'negative']:
                pipe_basic_outputs[name] = bus_outputs[name]

        # _CONF.ensure_required_parameters(bus_outputs)
        _CONF.handle_special_parameters(bus_outputs)

        # Prepare and return the output bus tuple with updated values
        out_bus = tuple(bus_outputs[name] for name in ProfileNodeUniversal.INPUT_NAMES)
        out_pipe_basic = tuple(pipe_basic_outputs[name] for name in ProfileNodePipeBasic.INPUT_NAMES)
        
        return (out_bus,) + (out_pipe_basic,) + out_bus
