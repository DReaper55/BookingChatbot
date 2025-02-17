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


def load_t5_model_and_tokenizer(from_saved=False):
    from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model and tokenizer
    MODEL_NAME = "google-t5/t5-small"  # Can use "t5-base" or "t5-large" for better performance

    if from_saved:
        MODEL_NAME = get_path_to(AssetPaths.T5_MODEL.value)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return model, tokenizer, data_collator
