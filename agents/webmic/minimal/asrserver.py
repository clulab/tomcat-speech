#!/usr/bin/env python

import asyncio
import websockets
import logging
from logging import debug, info


async def process_audio(message):
    info(f"Received {len(message)} bytes from browser.")


async def message_handler(websocket, path):
    async for message in websocket:
        await process_audio(message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_server = websockets.serve(message_handler, "127.0.0.1", 9000)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
