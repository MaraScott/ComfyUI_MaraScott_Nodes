#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

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
        for input in inputs:
            print(input)

        return web.json_response(
            {"message": f"inputs"}
        )

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')
