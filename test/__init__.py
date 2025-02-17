import json

json_file = "D:/Development/PycharmProjects/BookingChatbot/data/raw/train/dialogues_001.json"

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(json.dumps(data, indent=2))  # Pretty print the JSON
