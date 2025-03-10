import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer
import torch

from src.utils.singleton_meta import SingletonMeta


class SlotExtraction(metaclass=SingletonMeta):
    """Generates classified intents responses using a fine-tuned T5 model."""
    def __init__(self):
        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_SLOT_EXTRACTION_MODEL.value)
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model.to(self.__device)

    def generate_response(self, user_input):
        inputs = self.__tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=100, num_beams=5)

        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response


# user_query = "User: I'm looking for a handbag. Bot: Do you have a preferred brand? User: Michael Kors. Bot: What color do you prefer? User: brown. Bot: What material or style do you prefer? User: canvas. Bot: I found several Michael Kors handbags in brown with canvas style. What's your budget? User: Under $250."
#
# response = SlotExtraction().generate_response(user_query)
# print("Slots:", response)
