from typing import Dict

from fastapi import WebSocket

from fastapi import WebSocket
from typing import Dict

class WebSocketManager:
    """Manages WebSocket connections for multiple users."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}  # Maps user_id to WebSocket

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accepts and stores a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        """Removes a disconnected WebSocket connection."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_message(self, user_id: str, message: str):
        """Sends a message to a specific user if they are connected."""
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)
