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


# ..............................................
# Process the data.txt dataset.
# Convert it to a json
# ..............................................
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


# ..............................................
# Process the RAG dataset.
# The output for buy_product should specifically
# use "buy" or "purchase" in it's sentence
# instead of "I'm looking for"
# ..............................................
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

# ..............................................
# Process the context_translator_dataset
# Join all the conversations to form an input
# ..............................................
def load_and_preprocess_data(file_path, output_path):
    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
        data = json.load(f)

    # Shuffle dataset for randomness
    random.shuffle(data)

    # Extract input (conversation) and output (structured task)
    # formatted_data = [{"input": " ".join(d["conversation"]), "output": d["structured_task"]} for d in data]
    # formatted_data = [{"input": " ".join(d["input"]), "output": d["output"]} for d in data]

    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)


# ..............................................
# Process the context_translator_dataset
# Create the slot-filler dataset by splitting
# the conversations to get a turn-based convo
# ..............................................
def process_conversations(input_file, output_file):
    with open(input_file, "r", encoding="utf-8", errors='ignore') as file:
        dataset = json.load(file)

    structured_data = []

    for entry in dataset:
        conversation = entry["conversation"]
        collected_exchanges = []

        for i in range(0, len(conversation) - 1, 2):  # Process user-bot pairs
            collected_exchanges.append(conversation[i])  # User input
            if i + 1 < len(conversation):  # Ensure there is a bot response
                bot_response = conversation[i + 1]
                structured_data.append({
                    "input": collected_exchanges.copy(),
                    "output": bot_response.replace("Bot: ", "")
                })

    with open(output_file, "w", encoding="utf-8") as output_file:
        json.dump(structured_data, output_file, indent=2, ensure_ascii=False)

    print(f"Processed data saved to {output_file}")


# process_conversations(get_path_to(AssetPaths.RAW_CONTEXT_TRANSLATOR_DATASET.value), get_path_to(AssetPaths.SLOT_FILLER_DATASET.value))

load_and_preprocess_data(get_path_to(AssetPaths.RAW_FEATURE_EXTRACTION_DATASET.value), get_path_to(AssetPaths.FEATURE_EXTRACTION_DATASET.value))

# modify_rag_for_buyproduct_json(get_path_to(AssetPaths.RAW_RAG_DATASET.value))

# reformat_records(get_path_to(AssetPaths.RAW_SYNTHETIC_DATASET.value))

# format_dataset(dataset_dir="test", output_path=AssetPaths.TEST_DATASET.value)
