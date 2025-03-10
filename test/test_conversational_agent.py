from src.agents.conversational_agent import ConversationalAgent
from src.repository.opensearch_query_service import ProductsRetrievalService
from src.services.state_service import StateService, ConversationStates

user_id = "67c76b5be5b4aa75082eed96"
chat_id = "124"

ProductsRetrievalService().sync_mongo_to_opensearch()

agent = ConversationalAgent()

state_service = StateService()
state_service.update_state(user_id, ConversationStates.CONTEXT_TRANSLATOR.value)

response = agent.handle_user_message(user_id, chat_id, "I don't have a specific material or style")

print(f'Response: {response}')
