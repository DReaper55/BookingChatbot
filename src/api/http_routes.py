from fastapi import APIRouter
from pydantic import BaseModel

from src.agents.conversational_agent import ConversationalAgent

router = APIRouter()
agent = ConversationalAgent()  # Keep a running instance

class UserInput(BaseModel):
    user_id: str
    message: str

@router.post("/chat")
def chat(user: UserInput):
    response = agent.handle_user_message(user.user_id, user.message)
    return {"response": response}

@router.get("/")
def home():
    return {"message": "Conversational Agent is running!"}