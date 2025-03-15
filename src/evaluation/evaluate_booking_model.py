import os
import sys

from src.utils.asset_paths import AssetPaths

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.env_keys import EnvKeys
from src.utils.helpers import reformat_text
import torch

from src.utils.singleton_meta import SingletonMeta

from dotenv import load_dotenv
load_dotenv()

class BookingModel(metaclass=SingletonMeta):
    """Generates classified intents responses using a fine-tuned T5 model."""
    def __init__(self):
        # self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_BOOKING_MODEL.value)
        self.__model, self.__tokenizer = None, None
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_response(self, user_input, active_intent="NONE", slot_values="NONE", retrieved="NONE"):
        """
        Generates a response based on user input, intent, and slot values.
        """
        from src.utils.helpers import load_t5_model_and_tokenizer

        # self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, os.getenv(EnvKeys.RAG_BASED_BOOKING_MODEL.value))
        # self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_DISTIL_BOOKING_MODEL.value)
        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_BOOKING_MODEL.value)
        self.__model.to(self.__device)

        # Format input for T5
        input_text = f"generate response: {user_input}. Intent: {active_intent}. Slots: {slot_values}. Retrieved: {retrieved}"

        input_text = reformat_text(input_text)

        print(f'augmented_input 2: {input_text}')

        # Tokenize input
        inputs = self.__tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=128, num_beams=5)

        # Decode output
        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response


# Example 1: Restaurant booking request
# user_query = "I'm looking for a local place to dine in the centre that serves Chinese food."
# active_intent = "find_restaurant"
# slot_values = {
#     "restaurant-area": ["centre"],
#     "restaurant-food": ["chinese"]
# }
#
# response = generate_response(user_query, active_intent, slot_values)
# print("Bot:", response)

# Example 2: Train booking request
# user_query = "Hello, I am looking for a restaurant in Cambridge. I believe it is called Golden Wok"
# active_intent = "find_product"
# slot_values = {
#     "restaurant-name": ["golden wok"],
# }
#
# response = generate_response(user_query, active_intent, slot_values)
# print("Bot:", response)

model = BookingModel()

user_input = "I'm looking for a denim jackets in size S"
active_intent="find_product"
slot_values="product-type=jacket, feature=denim, size=S"
retrieved="feature=denim, available=20, price=$45, size=S"

response = model.generate_response(user_input, active_intent, slot_values, retrieved)
print("Bot:", response)