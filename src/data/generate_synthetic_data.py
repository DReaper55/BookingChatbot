import random

from src.utils.user_queries_for_synthesis import get_train_queries, get_hotel_queries


# Function to create synthetic dataset
def generate_train_synthetic_data():
    user_queries, retrieved_data_samples = get_train_queries()

    dataset = []

    for query in user_queries:
        retrieved_info = random.choice(retrieved_data_samples)
        intent = "find_train"
        slots = [f"{k}={v}" for k, v in retrieved_info.items() if k in ["train-day", "train-departure", "train-destination"]]
        slots_str = ", ".join(slots)

        retrieved_str = ", ".join([f"{k}: {v}" for k, v in retrieved_info.items()])

        input_text = f"generate response: {query} Intent: {intent}. Slots: {slots_str}. Retrieved: {retrieved_str}"

        response = f"I found {retrieved_info.get('available_trains', 'some')} trains for you. "
        if "first_departure" in retrieved_info:
            response += f"The first train leaves at {retrieved_info['first_departure']} and the last one at {retrieved_info['last_departure']}. "
        if "average_price" in retrieved_info:
            response += f"The average price is {retrieved_info['average_price']}. "

        response += "Would you like to book one of these?"

        dataset.append({"input": input_text, "output": response})

    return dataset


# Function to create synthetic dataset
def generate_synthetic_data():
    user_queries, retrieved_data_samples = get_hotel_queries()

    dataset = []

    for query in user_queries:
        retrieved_info = random.choice(retrieved_data_samples)
        intent = "find_hotel"
        slots = [f"{k}={v}" for k, v in retrieved_info.items() if k in ["hotel-location", "check-in", "check-out", "hotel-preference"] and v is not None]
        slots_str = ", ".join(slots)

        retrieved_str = ", ".join([f"{k}: {v}" for k, v in retrieved_info.items()])

        input_text = f"generate response: {query} Intent: {intent}. Slots: {slots_str}. Retrieved: {retrieved_str}"

        response = f"I found {retrieved_info.get('available-hotels', 'some')} hotels for you. "
        if "first-available-check-in" in retrieved_info:
            response += f"The earliest check-in is {retrieved_info['first-available-check-in']} and the latest check-out is {retrieved_info['last-available-check-out']}. "
        if "average-price" in retrieved_info:
            response += f"The average price is {retrieved_info['average-price']}. "

        response += "Would you like to see more details?"

        dataset.append({"input": input_text, "output": response})

    return dataset


# Generate hotel dataset
# synthetic_dataset = generate_synthetic_data()


# Generate train dataset
# synthetic_dataset = generate_train_synthetic_data()

# Print examples
# for data in synthetic_dataset:
#     print(data, "\n")
