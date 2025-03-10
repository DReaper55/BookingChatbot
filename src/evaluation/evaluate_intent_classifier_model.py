from src.utils.asset_paths import AssetPaths
from src.utils.env_keys import EnvKeys
from src.utils.helpers import load_t5_model_and_tokenizer
import torch

from src.utils.singleton_meta import SingletonMeta

import os
from dotenv import load_dotenv
load_dotenv()

class IntentClassifier(metaclass=SingletonMeta):
    """Generates classified intents responses using a fine-tuned T5 model."""
    def __init__(self):
        # self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_INTENT_CLASSIFIER_MODEL.value)
        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, os.getenv(EnvKeys.INTENT_CLASSIFIER_MODEL.value))
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model.to(self.__device)

    def generate_response(self, user_input):
        inputs = self.__tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=50, num_beams=5)

        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response


# user_query = "Can you tell me about the Apple Watch Series 9?"
#
# response = IntentClassifier().generate_response(user_query)
# print("Intent:", response)
