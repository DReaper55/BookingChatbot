import os
import re

import torch

from src.utils.asset_paths import AssetPaths


def get_path_to(dir):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../'))
    return os.path.join(project_root, dir)


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9.,!?\'\s-]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


def extract_text_and_intent(data):
    input_text = data["input"]

    # Extract user utterance
    text_match = re.search(r"generate response:\s*(.*?)\s*Intent:", input_text, re.IGNORECASE)
    text = text_match.group(1) if text_match else ""

    # Extract intent
    intent_match = re.search(r"Intent:\s*([\w_]+)", input_text, re.IGNORECASE)
    intent = intent_match.group(1) if intent_match else ""

    return {"text": text, "intent": intent}


def load_t5_model_and_tokenizer(from_saved=False, model_path=""):
    from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model and tokenizer
    MODEL_NAME = "google-t5/t5-small"  # Can use "t5-base" or "t5-large" for better performance

    if from_saved:
        MODEL_NAME = get_path_to(model_path)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return model, tokenizer, data_collator
