#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from . import __SESSIONS_DIR__, __PROFILES_DIR__
from .py.nodes.bus import Bus_node

import os
import json
from aiohttp import web
from server import PromptServer

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MarasitBusNode": Bus_node,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarasitBusNode": "Bus Node"
}

if hasattr(PromptServer, "instance"):
    @PromptServer.instance.routes.post("/marasit/bus")
    async def set_entries(request):
        json_data = await request.json()
        inputs = json_data.get("inputs")
        profile = json_data.get("profile")
        sid = json_data.get("session_id")
        nid = json_data.get("node_id")

        filename = f"session_{sid}_node_{nid}.json"
        print(filename)
        filepath = os.path.join(__SESSIONS_DIR__, filename)
        # Write the data to the file
        with open(filepath, 'w') as file:
            json.dump(profile, file)

        filename = f"profile_{profile}.json"
        filepath = os.path.join(__PROFILES_DIR__, filename)
        # Write the data to the file
        with open(filepath, 'w') as file:
            json.dump(inputs, file)

        return web.json_response(
            {"message": f"profile: {profile} | id: {nid}"}
        )

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')
