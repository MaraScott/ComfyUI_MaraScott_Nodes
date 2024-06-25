#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

class Configuration:
    
    OUTPUT_NODE = False
    CATEGORY = "MaraScott/Upscaling"
    
    @classmethod
    def generate_entries(self, input_names, input_types, code = 'py'):
        entries = {}
        for name, type_ in zip(input_names, input_types):
            # Handle special cases where additional parameters are needed
            if code == 'js':
                entries[name] = type_
            else:
                entries[name] = (type_, {"multiline": True})
                    
        return entries
    
    def determine_output_value(name = '', input_value = None, preset_value = None):
        return input_value if input_value is not None else preset_value if preset_value is not None else ''        