from transformers import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5Tokenizer

from src.actions.retriever import Retriever
from src.agents.booking_agent import BookingAgent
from src.utils.asset_paths import AssetPaths
from src.utils.helpers import get_path_to


def load_model(model_path):
    """Load a fine-tuned T5 model from a given path."""
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer

# Load models
intent_model, intent_tokenizer = load_model(get_path_to(AssetPaths.T5_INTENT_CLASSIFIER_MODEL.value))
slot_model, slot_tokenizer = load_model(get_path_to(AssetPaths.T5_SLOT_EXTRACTION_MODEL.value))
response_model, response_tokenizer = load_model(get_path_to(AssetPaths.T5_BOOKING_MODEL.value)) # 128 max length model

agent = BookingAgent(intent_model, intent_tokenizer, slot_model, slot_tokenizer, response_model, response_tokenizer, Retriever())

# Example user input
user_input = "I want to buy a red shirt for dinner?"
response = agent.generate_response(user_input)
print(f"{response} \n\n")
