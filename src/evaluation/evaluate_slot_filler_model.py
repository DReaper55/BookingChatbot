from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer
import torch


model, tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_SLOT_FILLER_MODEL.value)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_response(user_input):
    """
    Generates a response based on user input, intent.
    """
    user_input = f"ask question: {user_input}"
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=128, num_beams=5)

    # Decode output
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


# user_query = "User: I'm looking for a handbag. Bot: Do you have a preferred brand? User: Michael Kors. Bot: What color do you prefer? User: brown. Bot: What material or style do you prefer? User: canvas. Bot: I found several Michael Kors handbags in brown with canvas style. What's your budget? User: Under $250."
user_query = "User: I'd like to get a new pair of shoes. Bot: Any specific brand preference? User: Maybe one from Nike. Bot: What size do you wear? User: Size 42 for men. Bot: Any price range you're considering? User: Something cheap would be nice."

response = generate_response(user_query)
print("Bot:", response)
