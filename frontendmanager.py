from aiohttp import web
import threading
import json
import asyncio
import os
import struct
import io
from PIL import Image

class ThreadAccessType:
    INPUTS = 0

clients_connected = set()
lock = threading.Lock()

input_queue = []

def message_received(message):
    with lock:
        global input_queue
        input_queue.append(json.loads(message)["input"])

async def frontend_handler(request):
    return web.FileResponse(f"frontend_static/frontend.html")

async def ws_handler(request):
    ws = web.WebSocketResponse()
    global clients_connected
    await ws.prepare(request)
    with lock:
        clients_connected.add(ws)
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            message_received(msg.data)
    with lock:
        clients_connected.remove(ws)
    return ws

async def send_news(data):
    for ws in clients_connected:
            if not ws.closed:
                await ws.send_bytes(data)

server_eloop: asyncio.AbstractEventLoop = None

def start_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    global server_eloop
    server_eloop = loop

    app = web.Application()
    app.router.add_get("/", frontend_handler)
    app.router.add_get("/stream", ws_handler)
    app.add_routes([web.static("/s", "frontend_static")])
    handler = app.make_handler()

    server = loop.create_server(handler, host="127.0.0.1", port=17890)
    loop.run_until_complete(server)
    loop.run_forever()

def get_event_loop():
    return server_eloop

def getInputQueue():
    a = input_queue.copy()
    input_queue.clear()
    return a

def start():
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    return server_thread