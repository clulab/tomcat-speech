#!/usr/bin/env python

import asyncio
import websockets
from logging import debug

f = open("audio.raw", "wb")
n = 0

async def consume(message):
    global n
    n = n+1
    if n >= 100:
        debug("finshed recording")
        f.close()
    else:
        debug("writing chunk")
        f.write(message)

async def consumer_handler(websocket, path):
    async for message in websocket:
        await consume(message)

start_server = websockets.serve(consumer_handler, "127.0.0.1", 9000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
