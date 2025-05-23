import subprocess
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

import sys
import os

from src.utils.env_keys import EnvKeys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from http_routes import router as http_router
from auth_routes import router as auth_router
from src.api.auth_middleware import auth_middleware_factory
from src.repository.opensearch_query_service import ProductsRetrievalService
from ws_routes import router as ws_router

# from src.repository.opensearch_query_service import ProductsRetrievalService

app = FastAPI()

# Register middleware
app.add_middleware(
    auth_middleware_factory(),
)

# Define the allowed origins
origins = [
    "http://localhost",
    "http://localhost:8000",
    os.getenv(EnvKeys.SERVER_URL.value),
    "http://localhost:3000",  # Client app
    os.getenv(EnvKeys.CLIENT_URL.value),  # Client app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for HTTP and WebSocket endpoints
app.include_router(http_router)
app.include_router(auth_router, prefix="/auth")
# app.include_router(ws_router)

# ProductsRetrievalService().sync_mongo_to_opensearch()

subprocess.run(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

# def start_server():
#     # Start Docker container (if not already running)
#     # try:
#     #     subprocess.run(["docker-compose", "up", "-d"], check=True)
#     #     print("Docker container started successfully.")
#     # except subprocess.CalledProcessError:
#     #     print("Failed to start Docker container. Ensure it exists.")
#
#     # Sync databases
#     ProductsRetrievalService().sync_mongo_to_opensearch()
#
#     # Start server
#     subprocess.run(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
#
# if __name__ == "__main__":
#     start_server()
