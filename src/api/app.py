import subprocess
from fastapi import FastAPI

from http_routes import router as http_router
from ws_routes import router as ws_router

from src.repository.opensearch_query_service import ProductsRetrievalService

app = FastAPI()

# Include routers for HTTP and WebSocket endpoints
app.include_router(http_router)
app.include_router(ws_router)

def start_server():
    # Start Docker container (if not already running)
    # try:
    #     subprocess.run(["docker-compose", "up", "-d"], check=True)
    #     print("Docker container started successfully.")
    # except subprocess.CalledProcessError:
    #     print("Failed to start Docker container. Ensure it exists.")

    # Sync databases
    ProductsRetrievalService().sync_mongo_to_opensearch()

    # Start server
    subprocess.run(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

if __name__ == "__main__":
    start_server()
