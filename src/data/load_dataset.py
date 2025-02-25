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


def preprocess_fn(examples, process_for_context=False):
    if process_for_context:
        inputs = [f"translate conversation: {conv}" for conv in examples["input"]]
    else:
        inputs = examples["input"]

    targets = examples["output"]

    # Tokenize inputs and outputs
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)

    # Add labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ..........................................
# Load and preprocess data for intent
# classification model
# ..........................................
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
        preprocess_fn,
        batched=True, batch_size=1000
    )
    dev_dataset = dev_dataset.map(
        preprocess_fn,
        batched=True, batch_size=1000
    )

    return train_dataset, dev_dataset



# ..........................................
# Load and preprocess data for slot
# extraction model
# ..........................................

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
def load_and_preprocess_data(path, split_dataset=False):
    dataset = load_dataset("json", data_files=path, split="train")

    processed_data = []
    for item in dataset:
        text_intent = extract_text_and_intent(item)
        slots = extract_slots(item)

        # Create examples for multi-task learning
        processed_data.append({
            "input": f"classify intent: {text_intent['input']}",
            "output": text_intent["output"]
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

    if split_dataset:
        # Calculate the lengths of the train and eval sets
        train_length = int(len(dataset) * .8)
        eval_length = len(dataset) - train_length

        # Split the dataset
        train_dataset, eval_dataset = random_split(dataset, [train_length, eval_length])

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

        return train_dataloader.dataset, eval_dataloader.dataset

    return dataset


# ..........................................
# Load and preprocess data for RAG-based
# training
# ..........................................
def load_rag_dataset(split_ratio=0.8, for_booking_finetune=True, for_intent_finetune=False, for_slot_finetune=False):
    dataset = None

    # Load local dataset in streaming mode
    if for_booking_finetune:
        dataset = load_dataset(
            "json",
            data_files=get_path_to(AssetPaths.SYNTHETIC_DATASET.value),
            split="train",
            streaming=False
        ).map(lambda example: {"input": reformat_text(example['input'])})

    if for_slot_finetune:
        dataset = load_dataset(
            "json",
            data_files=get_path_to(AssetPaths.SYNTHETIC_DATASET.value),
            split="train",
            streaming=False
        ).map(extract_slots)


    if for_intent_finetune:
        dataset = load_dataset(
            "json",
            data_files=get_path_to(AssetPaths.SYNTHETIC_DATASET.value),
            split="train",
            streaming=False
        ).map(extract_text_and_intent)


    if dataset is None:
        return

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


# ..........................................
# Load and preprocess data for context
# translation training
# ..........................................
def load_context_translation_dataset(split_ratio=.8):
    dataset = load_dataset(
        "json",
        data_files=get_path_to(AssetPaths.CONTEXT_TRANSLATOR_DATASET.value),
        split="train",
        streaming=False,
    ).map(
        lambda example: preprocess_fn(example, True),
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
