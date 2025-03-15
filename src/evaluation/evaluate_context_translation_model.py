import os
import sys

from src.utils.asset_paths import AssetPaths

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.env_keys import EnvKeys
import torch

from src.utils.singleton_meta import SingletonMeta

from dotenv import load_dotenv
load_dotenv()

class ContextTranslator(metaclass=SingletonMeta):
    """Generates classified intents responses using a fine-tuned T5 model."""
    def __init__(self):
        # self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_CONTEXT_TRANSLATOR_MODEL.value)
        self.__model, self.__tokenizer = None, None
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def __format_input(text):
        # Split the text into lines
        lines = text.splitlines()

        # Function to add a period if necessary
        def add_period_if_needed(sentence):
            if not sentence[-1] in ['.', '?', '!']:
                sentence += '.'
            return sentence

            # Process each line
        formatted_lines = []
        for line in lines:
            if line.startswith("User:"):
                parts = line.split("User:", 1)
                parts[1] = add_period_if_needed(parts[1].strip())
                formatted_lines.append("User: " + parts[1])
            else:
                formatted_lines.append(line)

        # Join the lines back into a single string
        formatted_query = "\n".join(formatted_lines)

        return formatted_query

    def generate_response(self, user_input):
        from src.utils.helpers import load_t5_model_and_tokenizer

        # self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, os.getenv(EnvKeys.CONTEXT_TRANSLATOR_MODEL.value))
        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_DISTIL_CONTEXT_TRANSLATOR_MODEL_2.value)
        # self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_CONTEXT_TRANSLATOR_MODEL.value)
        self.__model.to(self.__device)

        if type(user_input) is not str:
            user_input = "\n".join(user_input)

        user_input = self.__format_input(user_input)

        print(f'augmented_input 4: {user_input}')

        user_input = f"translate context: {user_input}"
        inputs = self.__tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=100, num_beams=5)

        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return response


user_query = "User: I'm looking for a handbag. Bot: Do you have a preferred brand? User: Michael Kors. Bot: What color do you prefer? User: brown. Bot: What material or style do you prefer? User: canvas. Bot: I found several Michael Kors handbags in brown with canvas style. What's your budget? User: Under $250."
# user_query = "User: I'd like to get a new pair of shoes. Bot: Any specific brand preference? User: Maybe one from Nike. Bot: What size do you wear? User: Size 42 for men. Bot: Any price range you're considering? User: Something cheap would be nice."
# user_query = """
# User: I'm looking for a shirt
# Bot: Do you have a preferred brand?
# User: Adidas
# Bot: What color do you prefer?
# User: Yellow
# Bot: What material or style do you prefer?
# User: In size XL
# """
#
# response = ContextTranslator().generate_response(user_query)
# print("Bot:", response)
