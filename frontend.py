from aiohttp import web
import asyncio
import threading

async def frontend_handler(request):
    return web.FileResponse(f"frontend_static/frontend.html")

def start_server():
    app = web.Application()
    app.router.add_get("/", frontend_handler)
    app.add_routes([web.static("/s", "frontend_static")])
    runner = web.AppRunner(app)
    asyncio.run(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", "17890")
    asyncio.run(site.start())
    asyncio.get_event_loop().run_forever()

def start():
    thr = threading.Thread(target=start_server)
    thr.daemon = True
    thr.start()
    return thr
