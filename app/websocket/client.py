import asyncio
import json
import logging
from datetime import datetime
from typing import Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

class OrderBookWebSocket:
    def __init__(self, uri: str, symbol: str, callback: Callable):
        self.uri = uri
        self.symbol = symbol
        self.callback = callback
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.last_message_time = None

    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            logger.info(f"Connected to WebSocket for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    async def subscribe(self):
        if not self.is_connected:
            return

        subscribe_message = {
            "type": "subscribe",
            "symbol": self.symbol
        }
        await self.websocket.send(json.dumps(subscribe_message))

    async def process_messages(self):
        while self.is_connected:
            try:
                message = await self.websocket.recv()
                self.last_message_time = datetime.utcnow()
                
                # Parse and process the message
                data = json.loads(message)
                await self.callback(data)
                
            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.is_connected = False
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def start(self):
        while True:
            if not self.is_connected:
                connected = await self.connect()
                if connected:
                    await self.subscribe()
                    await self.process_messages()
            
            # If disconnected, wait before attempting to reconnect
            await asyncio.sleep(5)

    async def close(self):
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False 