from datasets import load_dataset

from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer, get_path_to, extract_text_and_intent, \
    extract_slots

# Load T5 tokenizer
_, tokenizer, _ = load_t5_model_and_tokenizer()

# ..........................................
# Load and preprocess data for booking model
# ..........................................
def preprocess_booking_function(examples):
    inputs = examples["input"]
    targets = examples["output"]

    return tokenizer(
        inputs,
        text_target=targets,
        padding="max_length",
        truncation=True,
        max_length=128
    )

def load_booking_dataset():
    # Load local dataset in streaming mode
    train_dataset = load_dataset(
        "json",
        data_files=get_path_to(AssetPaths.TRAINING_DATASET.value),
        split="train",
        streaming=False
    ).map(preprocess_booking_function, batched=True, batch_size=1000)

    dev_dataset = load_dataset(
            "json",
            data_files=get_path_to(AssetPaths.VALIDATION_DATASET.value),
            split="train",
            streaming=False
    ).map(preprocess_booking_function, batched=True, batch_size=1000)


    return train_dataset, dev_dataset


# ..........................................
# Load and preprocess data for intent
# classification model
# ..........................................
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



# ..........................................
# Load and preprocess data for slot
# extraction model
# ..........................................
def preprocess_slot_extract_fn(examples):
    inputs = examples["input"]
    targets = examples["output"]

    # Tokenize inputs and outputs
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=32)

    # Add labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_slot_extraction_dataset():
    # Load local dataset in streaming mode
    train_dataset = load_dataset(
        "json",
        data_files=get_path_to(AssetPaths.TRAINING_DATASET.value),
        split="train",
        streaming=False
    ).map(extract_slots)

    dev_dataset = load_dataset(
            "json",
            data_files=get_path_to(AssetPaths.VALIDATION_DATASET.value),
            split="train",
            streaming=False
    ).map(extract_slots)

    # Preprocess on-the-fly
    train_dataset = train_dataset.map(
        preprocess_slot_extract_fn,
        batched=True, batch_size=1000
    )
    dev_dataset = dev_dataset.map(
        preprocess_slot_extract_fn,
        batched=True, batch_size=1000
    )

    return train_dataset, dev_dataset

