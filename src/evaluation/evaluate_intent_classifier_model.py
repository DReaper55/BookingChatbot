from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer
import torch


model, tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_INTENT_CLASSIFIER_MODEL.value)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_response(user_input):
    """
    Generates a response based on user input, intent, and slot values.
    """
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=100, num_beams=5)

    # Decode output
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


user_query = "Can you book me a table for 11:00 on Friday?"

response = generate_response(user_query)
print("Intent:", response)
