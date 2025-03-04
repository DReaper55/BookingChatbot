import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agents.conversational_agent import ConversationalAgent
from src.api.ws_manager import WebSocketManager

router = APIRouter()
agent = ConversationalAgent()

# Create a WebSocket manager instance
manager = WebSocketManager()

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_message(user_id, f"Received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(user_id)


@router.websocket("/ws/recommendations/{user_id}")
async def send_recommendations(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Fetch recommendations dynamically
            # recommendations = get_cf_recommendations(user_id)
            # await websocket.send_json(recommendations)
            await websocket.send_json("No recommendations found")
            await asyncio.sleep(5)  # Send updates every 5 seconds
    except WebSocketDisconnect:
        manager.disconnect(user_id)


@router.websocket("/ws/chat/{user_id}")
async def chat_websocket(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)

    try:
        while True:
            data = await websocket.receive_text()

            # Process user message
            response = agent.handle_user_message(user_id, data)

            # Send response back in real-time
            await manager.send_message(user_id, response)

    except WebSocketDisconnect:
        manager.disconnect(user_id)
