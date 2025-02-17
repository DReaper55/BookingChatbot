from src.utils.helpers import load_t5_model_and_tokenizer
import torch


model, tokenizer, _ = load_t5_model_and_tokenizer(True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_response(user_input, active_intent="NONE", slot_values={}):
    """
    Generates a response based on user input, intent, and slot values.
    """
    # Format input for T5
    slot_values_str = ", ".join([f"{k}={', '.join(v)}" for k, v in slot_values.items()])
    input_text = f"generate response: {user_input} Active intent: {active_intent}. Slot values: {slot_values_str}."

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=100, num_beams=5)

    # Decode output
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
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
user_query = "I need train reservations from norwich to cambridge."
active_intent = "find_train"
slot_values = {
    "train-departure": ["Norwich"],
    "train-destination": ["Cambridge"],
    "train-day": ["Monday"],
    "train-arriveby": ["18:00"],
    "train-leaveat": ["8:00"],
}

response = generate_response(user_query, active_intent, slot_values)
print("Bot:", response)
