from datasets import load_dataset

from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer, get_path_to, extract_text_and_intent

# Load T5 tokenizer
_, tokenizer, _ = load_t5_model_and_tokenizer()

def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]

    return tokenizer(
        inputs,
        text_target=targets,
        padding="max_length",
        truncation=True,
        max_length=128
    )


def load_t5_dataset():
    # Load local dataset in streaming mode
    train_dataset = load_dataset(
        "json",
        data_files=get_path_to(AssetPaths.TRAINING_DATASET.value),
        split="train",
        streaming=False
    ).map(preprocess_function, batched=True, batch_size=1000)

    dev_dataset = load_dataset(
            "json",
            data_files=get_path_to(AssetPaths.VALIDATION_DATASET.value),
            split="train",
            streaming=False
    ).map(preprocess_function, batched=True, batch_size=1000)


    return train_dataset, dev_dataset


def preprocess_intent_class_fn(examples):
    inputs = examples["text"]
    targets = examples["intent"]

    # Tokenize inputs and outputs
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=32)

    # Add labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_intent_classifier_dataset():
    # Load local dataset in streaming mode
    train_dataset = load_dataset(
        "json",
        data_files=get_path_to(AssetPaths.TRAINING_DATASET.value),
        split="train",
        streaming=False
    ).map(extract_text_and_intent)

    dev_dataset = load_dataset(
            "json",
            data_files=get_path_to(AssetPaths.VALIDATION_DATASET.value),
            split="train",
            streaming=False
    ).map(extract_text_and_intent)

    # Preprocess on-the-fly
    train_dataset = train_dataset.map(
        preprocess_intent_class_fn,
        batched=True, batch_size=1000
    )
    dev_dataset = dev_dataset.map(
        preprocess_intent_class_fn,
        batched=True, batch_size=1000
    )

    return train_dataset, dev_dataset

