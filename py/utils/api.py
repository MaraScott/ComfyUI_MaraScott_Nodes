from aiohttp import web
from server import PromptServer

class Api :
    
    def __init__(self):

        app = web.Application()
        routes = web.RouteTableDef()

        @routes.get("/test")
        async def test(request):
            testy = request.rel_url.query.get("testy", "")

            test = False
            if testy == "test":
                test = True

            json_obj = { "name": "test", "title": "AnyBus - test" }

            return web.json_response(json_obj, content_type='application/json')

        app.add_routes(routes) 

        PromptServer.instance.app.add_subapp('/marascott', app)
