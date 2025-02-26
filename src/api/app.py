import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.agents.conversational_agent import ConversationalAgent

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)