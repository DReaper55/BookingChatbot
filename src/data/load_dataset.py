from datasets import load_dataset, Dataset
from torch.utils.data import random_split, DataLoader

from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer, get_path_to, extract_text_and_intent, \
    extract_slots, reformat_text

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
def preprocess_fn(examples):
    inputs = examples["input"]
    targets = examples["output"]

    # Tokenize inputs and outputs
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)

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
        preprocess_fn,
        batched=True, batch_size=1000
    )
    dev_dataset = dev_dataset.map(
        preprocess_fn,
        batched=True, batch_size=1000
    )

    return train_dataset, dev_dataset


# ..........................................
# Load and preprocess data for multi-task
# model
# ..........................................
def load_and_preprocess_data(path):
    dataset = load_dataset("json", data_files=path, split="train")

    processed_data = []
    for item in dataset:
        text_intent = extract_text_and_intent(item)
        slots = extract_slots(item)

        # Create examples for multi-task learning
        processed_data.append({
            "input": f"classify intent: {text_intent['text']}",
            "output": text_intent["intent"]
        })
        processed_data.append({
            "input": f"extract slots: {slots['input']}",
            "output": slots["output"]
        })
        processed_data.append({
            "input": item["input"],
            "output": item["output"]
        })

    dataset = Dataset.from_list(processed_data)

    dataset = dataset.map(
        preprocess_fn,
        batched=True, batch_size=1000
    )

    return dataset


# ..........................................
# Load and preprocess data for RAG-based
# training
# ..........................................
def load_rag_dataset(split_ratio=0.8):
    # Load local dataset in streaming mode
    dataset = load_dataset(
        "json",
        data_files=get_path_to(AssetPaths.SYNTHETIC_DATASET.value),
        split="train",
        streaming=False
    ).map(lambda example: {"input": reformat_text(example['input'])})

    # Preprocess on-the-fly
    dataset = dataset.map(
        preprocess_fn,
        batched=True, batch_size=500
    )

    # Calculate the lengths of the train and eval sets
    train_length = int(len(dataset) * split_ratio)
    eval_length = len(dataset) - train_length

    # Split the dataset
    train_dataset, eval_dataset = random_split(dataset, [train_length, eval_length])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    return train_dataloader.dataset, eval_dataloader.dataset
