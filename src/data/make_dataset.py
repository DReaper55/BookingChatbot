import json
import os
import random

from src.utils.asset_paths import AssetPaths
from src.utils.helpers import get_path_to


def preprocess_dialogues(dialogues):
    data_pairs = []

    for dialogue in dialogues:
        turns = dialogue["turns"]

        for i in range(len(turns) - 1):  # Ignore last turn if it's a user turn
            user_turn = turns[i]
            system_turn = turns[i + 1]

            if user_turn["speaker"] != "USER" or system_turn["speaker"] != "SYSTEM":
                continue  # Ensure it's a user-system pair

            # Extract user utterance
            user_text = user_turn["utterance"]

            # Extract intent & slots
            active_intent = "None"
            slot_values = {}

            for frame in user_turn["frames"]:
                if frame["state"]["active_intent"] != "NONE":
                    active_intent = frame["state"]["active_intent"]
                    slot_values.update(frame["state"]["slot_values"])

            # Format slots into key-value pairs
            slot_str = ", ".join([f"{k}={', '.join(v)}" for k, v in slot_values.items()])

            # Construct input prompt for T5
            input_text = f"generate response: {user_text}. Intent: {active_intent}. Slots: {slot_str if slot_str else 'None'}"

            # Extract system response
            system_text = system_turn["utterance"]

            # Append to dataset
            data_pairs.append({"input": input_text, "output": system_text})

    return data_pairs


def format_dataset(dataset_dir="test", output_path="output.json"):
    def load_json_files(directory):
        data = []
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    dialogues = json.load(f)
                    data.extend(dialogues)  # Ensure it's a list
        return data

    data_path = get_path_to('data/raw')

    # Load train, dev, test datasets
    data = load_json_files(os.path.join(data_path, dataset_dir))

    # Preprocess data
    formatted_data = preprocess_dialogues(data)

    # Save as JSON
    with open(get_path_to(output_path), "w") as f:
        json.dump(formatted_data, f, indent=2)

    print(f"Processed {len(formatted_data)} dialogue pairs.")


def reformat_records(file_path):
    records = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        current_record = {}
        for line in file:
            if line.startswith("Input:"):
                if current_record:
                    records.append(current_record)
                current_record = {"input": line[len("Input:"):].strip()}
            elif line.startswith("Output:"):
                current_record["output"] = line[len("Output:"):].strip()
        if current_record:
            records.append(current_record)

    with open(get_path_to(AssetPaths.SYNTHETIC_DATASET.value), 'w', encoding='utf-8') as json_file:
        json.dump(records, json_file, indent=4)


def modify_rag_for_buyproduct_json(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        data = json.load(file)

    for entry in data:
        input_text = entry.get("input", "")
        if "Intent: buy_product" in input_text and "generate response: I'm looking for" in input_text:
            choice = random.choice(["I want to buy", "I want to purchase"])
            new_input_text = input_text.replace("generate response: I'm looking for", f"generate response: {choice}")
            entry["input"] = new_input_text

    with open(get_path_to(AssetPaths.PROCESSED_RAG_DATASET.value), 'w') as file:
        json.dump(data, file, indent=4)


modify_rag_for_buyproduct_json(get_path_to(AssetPaths.RAW_RAG_DATASET.value))

# reformat_records(get_path_to(AssetPaths.RAW_SYNTHETIC_DATASET.value))

# format_dataset(dataset_dir="test", output_path=AssetPaths.TEST_DATASET.value)
