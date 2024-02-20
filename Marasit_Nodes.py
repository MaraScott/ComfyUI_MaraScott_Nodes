#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from . import __SESSIONS_DIR__, __PROFILES_DIR__
from .py.nodes.UniversalBusNode import UniversalBusNode
from .py.inc.profiles.default import Node as default
from .py.inc.profiles.pipe_basic import Node as pipe_basic
from .py.inc.profiles.pipe_detailer import Node as pipe_Detailer
from .py.nodes.Bus_node import Bus_node
from .py.nodes.BusNode import BusNode
from .py.nodes.AnyBusNode import AnyBusNode
from .py.nodes.PipeNodeBasic import PipeNodeBasic
from .py.nodes.BusPipeNode import BusPipeNode

import os
import json
from aiohttp import web
import importlib
from server import PromptServer

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MarasitBusNode": Bus_node,
    "MarasitAnyBusNode": AnyBusNode,
    # "MarasitBusNode": BusNode,
    # "MarasitBusPipeNode": BusPipeNode,
    # "MarasitPipeNodeBasic": PipeNodeBasic,
    # "MarasitUniversalBusNode": UniversalBusNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarasitBusNode": "Bus Node (OBSOLETE - use the universal bus instead)",
    "MarasitAnyBusNode": "AnyBus Node",
    # "MarasitBusNode": "Bus Node - Simple",
    # "MarasitBusPipeNode": "Bus/Pipe Node",
    # "MarasitPipeNodeBasic": "BasicPipe Node",
    # "MarasitUniversalBusNode": "Universal Bus Node"
}

if hasattr(PromptServer, "instance"):
    @PromptServer.instance.routes.post("/marasit/bus/profile")
    async def getNodeProfileEntries(request):
        json_data = await request.json()
        profile = json_data.get("profile")
        module_path = 'ComfyUI-MarasIT-Nodes.py.inc.profiles.' + profile  # Adjust the module path as needed
        module = importlib.import_module(module_path)
        NodeClass = getattr(module, 'Node')
        return web.json_response(
            {"entries": NodeClass.ENTRIES_JS}
        )
        

    @PromptServer.instance.routes.post("/marasit/bus/node/update")
    async def setNodeProfileEntries(request):
        json_data = await request.json()
        inputs = json_data.get("inputs")
        profile = json_data.get("profile")
        sid = json_data.get("session_id")
        nid = json_data.get("node_id")

        filename = f"session_{sid}_node_{nid}.json"
        filepath = os.path.join(__SESSIONS_DIR__, filename)
        # Write the data to the file
        with open(filepath, 'w') as file:
            json.dump(profile, file)

        filename = f"profile_{profile}.json"
        filepath = os.path.join(__PROFILES_DIR__, filename)

        if len(inputs) == 0:
            # Load inputs from the file
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    inputs = json.load(file)
            else:
                inputs = default.ENTRIES
        else:
            # Write the data to the file
            with open(filepath, 'w') as file:
                json.dump(inputs, file)

        return web.json_response(
            {"message": f"Node id {nid} and profile {profile} have been set"}
        )
        
    @PromptServer.instance.routes.post("/marasit/bus/profile")
    async def setDefaultProfile(request):
        json_data = await request.json()
        profile = json_data.get("profile")

        filename = f"profile_{profile}.json"
        filepath = os.path.join(__PROFILES_DIR__, filename)

        inputs = default.ENTRIES
        # Write the data to the file
        with open(filepath, 'w') as file:
            json.dump(inputs, file)

        return web.json_response(
            {"message": f"Profile {profile} has been set/reset"}
        )

    @PromptServer.instance.routes.post("/marasit/bus/node/remove")
    async def removeNodeProfile(request):
        json_data = await request.json()
        sid = json_data.get("session_id")
        nid = json_data.get("node_id")

        filename = f"session_{sid}_node_{nid}.json"
        filepath = os.path.join(__SESSIONS_DIR__, filename)
        
        if os.path.exists(filepath):  # Check if the file exists to avoid errors
            os.remove(filepath)
            message = f"Node id {nid} has been removed"
        else:
            message = f"Node id {nid} has not been removed cause the file {filename} do not exists"

        return web.json_response(
            {"message": message}
        )        

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')
