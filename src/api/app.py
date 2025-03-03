import subprocess

from fastapi import FastAPI
from pydantic import BaseModel

from src.agents.conversational_agent import ConversationalAgent
from src.repository.opensearch_query_service import ProductsRetrievalService

app = FastAPI()
agent = ConversationalAgent()  # Keep a running instance

class UserInput(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
def chat(user: UserInput):
    response = agent.handle_user_message(user.user_id, user.message)
    return {"response": response}

@app.get("/")
def home():
    return {"message": "Conversational Agent is running!"}

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
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    subprocess.run(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

if __name__ == "__main__":
    start_server()
