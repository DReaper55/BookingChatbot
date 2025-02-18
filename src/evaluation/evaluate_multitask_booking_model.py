from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer
import torch


model, tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_MULTITASK_BOOKING_MODEL.value)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def predict_intent_slot_response(text):
    inputs = [
        f"classify intent: {text}",
        f"extract slots: {text}",
        f"generate response: {text}"
    ]

    inputs_tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs_tokenized["input_ids"].to(model.device)

    outputs = model.generate(input_ids)
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return {"intent": predictions[0], "slots": predictions[1], "response": predictions[2]}


user_input = "Hello, I am looking for a restaurant in Cambridge. I believe it is called Golden Wok"
# user_input = "I need a train from Norwich to Cambridge on Monday."
result = predict_intent_slot_response(user_input)
print(result)
