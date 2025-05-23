import json
import os
import re
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.asset_paths import AssetPaths


def get_path_to(dir):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../'))
    return os.path.join(project_root, dir)


# ......................................
# Extract text and intent from an example
# ......................................
def extract_text_and_intent(data):
    input_text = data["input"]

    # Extract user utterance
    text_match = re.search(r"generate response:\s*(.*?)\s*Intent:", input_text, re.IGNORECASE)
    text = text_match.group(1) if text_match else ""

    # Extract intent
    intent_match = re.search(r"Intent:\s*([\w_]+)", input_text, re.IGNORECASE)
    intent = intent_match.group(1) if intent_match else ""

    return {"input": text, "output": intent}


# ......................................
# Extract text and slots from an example
# ......................................
def extract_slots(data):
    # Extract user input after "generate response:"
    input_text = re.search(r"generate response:\s*(.*?)\s*Intent:", data["input"])
    input_text = input_text.group(1).strip() if input_text else ""

    # Extract slot-value pairs
    slots_match = re.search(r"Slots:\s*(.*?)(?:Retrieved:|$)", data["input"])
    slots_text = slots_match.group(1).strip() if slots_match else ""

    # Convert slots into dictionary
    slots = {}
    for slot in slots_text.split(", "):
        if "=" in slot:
            key, value = slot.split("=", 1)
            key, value = key.strip(), value.strip()

            # Only add slot if its value appears in the input text
            if value.lower() in input_text.lower():
                slots[key] = value

    # Convert slots to output format
    slot_output = ", ".join([f"{k}={v}" for k, v in slots.items()])

    # Add to processed dataset
    return {
        "input": input_text,
        "output": slot_output
    }

# ......................................
# Format input text for training
# ......................................
def reformat_text(input_text):
        # Extract user input
        user_input_match = re.search(r"generate response:\s*(.*?)\s*Intent:", input_text)
        user_input = user_input_match.group(1).strip() if user_input_match else ""

        # Extract intent
        intent_match = re.search(r"Intent:\s*([\w_]+)", input_text)
        intent = intent_match.group(1).strip() if intent_match else ""

        # Extract slots
        slots_match = re.search(r"Slots:\s*(.*?)(?:Retrieved:|$)", input_text, re.DOTALL)
        slots_text = slots_match.group(1).strip() if slots_match else ""
        slots = [f"- {slot.strip()}" for slot in slots_text.split(", ") if "=" in slot]

        # Extract retrieved information
        retrieved_match = re.search(r"Retrieved:\s*(.*)", input_text)
        retrieved_text = retrieved_match.group(1).strip() if retrieved_match else ""

        # Process retrieved information
        retrieved_info = []
        retrieved_pairs = re.findall(r"([\w-]+)=([^,]+?)(?=,\s[\w-]+=|$)", retrieved_text)  # Ensure correct value capture

        for key, value in retrieved_pairs:
            value = value.strip()
            formatted_key = key.replace("-", " ").replace("_", " ").title()  # Convert "train-day" to "Train Day"
            value = value if value.lower() != "none" else "None (missing)"  # Handle missing values
            retrieved_info.append(f"- {formatted_key}: {value}")

        # Construct formatted text
        formatted_text = f"generate response: {user_input}\n"
        formatted_text += f"Intent: {intent}\n\n"
        formatted_text += "Slots:\n" + "\n".join(slots) + "\n\n" if slots else "Slots: None\n\n"
        formatted_text += "Retrieved Information:\n" + "\n".join(retrieved_info) if retrieved_info else '- None'

        return formatted_text


# ......................................
# Format search result to model's input
# ......................................
def format_extracted_features(input_string, is_json=False):
    product_info = input_string

    if not is_json:
        # Add braces to make it a valid dictionary string
        input_string = "{" + input_string + "}"

        # Fix missing double quotes around keys
        input_string = re.sub(r'(\w+):', r'"\1":', input_string)

        # Fix missing double quotes around unquoted string values (words without spaces)
        input_string = re.sub(r': (\w+)([,}])', r': "\1"\2', input_string)

        product_info = json.loads(input_string)

    # Remove None and 'null' values from filters
    product_info = {k: v for k, v in product_info.items() if v and v != 'null'}

    # Replace whitespaces with hyphens in the features list
    if "features" in product_info:
        product_info["features"] = [feature.replace(" ", "-") for feature in product_info["features"]]

    formatted_info = []

    # Iterate through the dictionary items
    for key, value in product_info.items():
        if key == "features":
            # For features, add each feature separately
            for feature in value:
                if is_json:
                    formatted_info.append(f"feature={feature}")
                else:
                    formatted_info.append(f"feature-{feature}={feature}")
        else:
            # For other key-value pairs, add them directly
            formatted_info.append(f"{key}={value}")

    # Join the formatted key-value pairs with commas
    return ", ".join(formatted_info)


# ......................................
# Load model and tokenizer
# ......................................
def load_t5_model_and_tokenizer(from_saved=False, model_path=None):
    from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, DataCollatorForSeq2Seq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model and tokenizer
    MODEL_NAME = model_path or "google-t5/t5-small"  # Can use "t5-base" or "t5-large" for better performance

    if from_saved:
        MODEL_NAME = get_path_to(model_path)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.half() # convert to FP16

    # model.push_to_hub("DReaper/feature-extraction-large")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return model, tokenizer, data_collator


# ......................................
# Upload a model to huggingface
# ......................................
def upload_model_to_huggingface():
    from huggingface_hub import HfApi

    # login()  # Logs you into Hugging Face
    api = HfApi()

    # Upload your model to your Hugging Face account
    api.upload_folder(
        folder_path=get_path_to(AssetPaths.T5_MULTITASK_FEATURE_EXTRACTION_MODEL.value),
        repo_id="DReaper/feature-extraction-large",
        repo_type="model"
    )


load_t5_model_and_tokenizer(True, get_path_to(AssetPaths.T5_MULTITASK_FEATURE_EXTRACTION_MODEL.value))

# upload_model_to_huggingface()
