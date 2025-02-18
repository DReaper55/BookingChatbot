from transformers import T5Tokenizer, T5ForConditionalGeneration

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
response_model, response_tokenizer = load_model(get_path_to(AssetPaths.T5_BOOKING_MODEL.value))

def generate_intent(user_input):
    """Predict intent from user input."""
    input_text = user_input
    inputs = intent_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    output = intent_model.generate(**inputs)
    intent = intent_tokenizer.decode(output[0], skip_special_tokens=True)
    return intent

def extract_slots(user_input):
    """Extract slot values from user input."""
    input_text = user_input
    inputs = slot_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    output = slot_model.generate(**inputs)
    slots = slot_tokenizer.decode(output[0], skip_special_tokens=True)
    return slots

def generate_response(user_input, intent, slots):
    """Generate a response based on user input, intent, and extracted slots."""
    input_text = f"generate response: {user_input} Intent: {intent} Slots: {slots}"
    inputs = response_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    output = response_model.generate(**inputs)
    response = response_tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def agent_pipeline(user_input):
    """Main agent function that processes a user input and returns a response."""
    intent = generate_intent(user_input)
    print(f"Intent: {intent}")
    slots = extract_slots(user_input)
    print(f"Slots: {slots}")
    response = generate_response(user_input, intent, slots)
    return response

# Example usage
# user_input = "I need a train from Norwich to Cambridge on Monday."
user_input = "Hello, I am looking for a restaurant in Cambridge. I believe it is called Golden Wok"
response = agent_pipeline(user_input)
print(f"Response: {response}")
