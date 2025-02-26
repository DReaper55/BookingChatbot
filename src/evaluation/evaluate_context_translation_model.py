from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer
import torch

from src.utils.singleton_meta import SingletonMeta


class ContextTranslator(metaclass=SingletonMeta):
    """Generates classified intents responses using a fine-tuned T5 model."""
    def __init__(self):
        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_CONTEXT_TRANSLATOR_MODEL.value)
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model.to(self.__device)

    def generate_response(self, user_input):
        if type(user_input) is not str:
            user_input = "\n".join(user_input)

        user_input = f"translate context: {user_input}"
        inputs = self.__tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=100, num_beams=5)

        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response


# user_query = "User: I'm looking for a handbag. Bot: Do you have a preferred brand? User: Michael Kors. Bot: What color do you prefer? User: brown. Bot: What material or style do you prefer? User: canvas. Bot: I found several Michael Kors handbags in brown with canvas style. What's your budget? User: Under $250."
# user_query = "User: I'd like to get a new pair of shoes. Bot: Any specific brand preference? User: Maybe one from Nike. Bot: What size do you wear? User: Size 42 for men. Bot: Any price range you're considering? User: Something cheap would be nice."
#
# response = ContextTranslator().generate_response(user_query)
# print("Bot:", response)
