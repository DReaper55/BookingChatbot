import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.asset_paths import AssetPaths
from src.utils.env_keys import EnvKeys
from src.utils.helpers import load_t5_model_and_tokenizer
import torch

from src.utils.singleton_meta import SingletonMeta

from dotenv import load_dotenv
load_dotenv()

class FeatureExtraction(metaclass=SingletonMeta):
    """Generates classified intents responses using a fine-tuned T5 model."""
    def __init__(self):
        # self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_MULTITASK_FEATURE_EXTRACTION_MODEL.value)
        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, os.getenv(EnvKeys.FEATURE_EXTRACTION_MODEL.value))
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model.to(self.__device)

    def extract_slot(self, text):
        text = f"extract slot: {text}"
        inputs = self.__tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=100, num_beams=5)

        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

    def retrieve_category(self, text):
            text = f"retrieve category: {text}"
            inputs = self.__tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.__device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.__model.generate(**inputs, max_length=50, num_beams=5)

            response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return response

    def extract_features(self, text):
            text = f"extract features: {text}"
            inputs = self.__tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.__device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.__model.generate(**inputs, max_length=128, num_beams=5)

            response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return response



# user_input = "I want to get a green shirt in medium size that's for biking"
user_input = "Buy a yellow casual dress in extra-large size for $120 with PayPal and standard delivery."
# result = extract_features(user_input)
result = FeatureExtraction().extract_features(user_input)
print(result)
