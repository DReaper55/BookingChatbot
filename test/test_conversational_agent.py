from src.agents.conversational_agent import ConversationalAgent

user_id = "user_125"

agent = ConversationalAgent()

response = agent.handle_user_message(user_id, "I don't have a specific material or style")

print(f'Response: {response}')
