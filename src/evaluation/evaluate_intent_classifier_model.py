import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.env_keys import EnvKeys
import torch

from src.utils.singleton_meta import SingletonMeta

from dotenv import load_dotenv
load_dotenv()

class IntentClassifier(metaclass=SingletonMeta):
    """Generates classified intents responses using a fine-tuned T5 model."""
    def __init__(self):
        # self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_INTENT_CLASSIFIER_MODEL.value)
        self.__model, self.__tokenizer = None, None
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_response(self, user_input):
        from src.utils.helpers import load_t5_model_and_tokenizer

        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, os.getenv(EnvKeys.INTENT_CLASSIFIER_MODEL.value))
        self.__model.to(self.__device)

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
