from datasets import load_dataset

from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer, get_path_to

# Load T5 tokenizer
_, tokenizer, _ = load_t5_model_and_tokenizer()

def format_and_preprocess_function(examples):
    inputs = []
    targets = []

    for example in examples["turns"]:
        for i in range(len(example) - 1):
            user_turn = example[i]
            system_turn = example[i + 1]

            if user_turn["speaker"] == "USER" and system_turn["speaker"] == "SYSTEM":
                active_intent = ""
                slot_values = []

                for frame in user_turn["frames"]:
                    if frame["state"]["active_intent"] != "NONE":
                        active_intent = frame["state"]["active_intent"]
                        for slot, values in frame["state"]["slot_values"].items():
                            slot_values.append(f"{slot}={', '.join(values)}")

                slot_values_str = ", ".join(slot_values)
                input_text = f"generate response: {user_turn['utterance']} Active intent: {active_intent}. Slot values: {slot_values_str}."
                target_text = system_turn["utterance"]

                inputs.append(input_text)
                targets.append(target_text)

    return tokenizer(inputs, text_target=targets, padding="max_length", truncation=True, max_length=128)


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


from datasets import Dataset, DatasetDict
import json
import os

def load_t5_dataset_new(data_path):
    def load_json_files(directory):
        data = []
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    dialogues = json.load(f)
                    data.extend(dialogues)  # Ensure it's a list
        return data

    # Load train, dev, test datasets
    train_data = load_json_files(os.path.join(data_path, "train"))
    dev_data = load_json_files(os.path.join(data_path, "dev"))
    test_data = load_json_files(os.path.join(data_path, "test"))

    # Convert to Hugging Face Dataset format
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "dev": Dataset.from_list(dev_data),
        "test": Dataset.from_list(test_data),
    })

    return dataset
