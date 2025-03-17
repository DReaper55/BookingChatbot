from src.agents.conversational_agent import ConversationalAgent
from src.repository.opensearch_query_service import ProductsRetrievalService
from src.services.state_service import StateService, ConversationStates

user_id = "012"
chat_id = "124"

# Synchronize mongodb with opensearch
# ProductsRetrievalService().sync_mongo_to_opensearch()

agent = ConversationalAgent()

state_service = StateService()

# Set model's state
state_service.update_state(user_id, ConversationStates.IDLE.value)

# Send a message to the agent
response = agent.handle_user_message(user_id, chat_id, "Hi")

print(f'Response: {response}')
