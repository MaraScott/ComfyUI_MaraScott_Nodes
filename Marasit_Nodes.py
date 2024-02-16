#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from . import __SESSIONS_DIR__, __PROFILES_DIR__
from .py.nodes.UniversalBusNode import UniversalBusNode, UniversalBusNodeProfiles
from .py.nodes.BusNode import BusNode

import os
import json
from aiohttp import web
from server import PromptServer

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MarasitBusNode": BusNode,
    "MarasitUniversalBusNode": UniversalBusNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarasitBusNode": "Bus Node",
    "MarasitUniversalBusNode": "Universal Bus Node"
}

if hasattr(PromptServer, "instance"):
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
                inputs = UniversalBusNodeProfiles.default
        else:
            # Write the data to the file
            with open(filepath, 'w') as file:
                json.dump(inputs, file)

        return web.json_response(
            {"message": f"Node id {nid} and profile {profile} have been set"}
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
