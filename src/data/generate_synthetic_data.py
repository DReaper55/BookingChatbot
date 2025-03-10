import random

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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


def generate_synthetic_find_product_data(num_samples=10):
    product_types = ["shirt", "shoe", "jacket", "accessory", "pants", "electronics", "dress", "toy", "book", "furniture"]
    features = ["red", "blue", "cotton", "leather", "waterproof", "wireless", "bodycon", "loose-fit", "gaming", "handmade"]
    sizes = ["XS", "S", "M", "L", "XL", "XXL"]
    brands = ["Nike", "Adidas", "Sony", "Apple", "Zara", "Gucci", "Ikea"]
    locations = ["Lagos", "New York", "London", "Paris", "Tokyo"]

    dataset = []

    for _ in range(num_samples):
        product_type = random.choice(product_types)
        selected_features = random.sample(features, k=random.randint(1, 3))
        size = random.choice(sizes) if product_type in ["shirt", "jacket", "pants", "dress", "shoe"] else None
        brand = random.choice(brands) if random.random() > 0.5 else None
        location = random.choice(locations) if random.random() > 0.7 else None
        available = random.randint(0, 20)
        price = f"${random.randint(20, 500)}"

        query = f"I want to buy a {' '.join(selected_features)} {product_type}"
        if size:
            query += f" in size {size}"
        if brand:
            query += f" from {brand}"
        if location:
            query += f" in {location}"

        input_text = f"generate response: {query}. Intent: find_product. Slots: product-type={product_type}, "
        input_text += ", ".join([f"feature={feature}" for feature in selected_features])
        if size:
            input_text += f", size={size}"

        retrieved_text = f"Retrieved: TYPE=<TYPE>, "
        retrieved_text += ", ".join(["FEATURE=<FEATURE>" for _ in selected_features])
        if size:
            retrieved_text += f", SIZE=<SIZE>"
        retrieved_text += f", AVAILABLE=<AVAILABLE>, PRICE=<PRICE>"
        if brand:
            retrieved_text += f", BRAND=<BRAND>"
        if location:
            retrieved_text += f", LOCATION=<LOCATION>"

        output_text = f"The <FEATURE> <TYPE> is available for <PRICE>. We have <AVAILABLE> in stock. Would you like to proceed with the purchase?"

        dataset.append({"input": input_text + " " + retrieved_text, "output": output_text})

    return dataset


# Generate RAG-based dataset for products
# data = generate_synthetic_find_product_data(5)
# for item in data:
#     print(item, "\n")


# Generate hotel dataset
# synthetic_dataset = generate_synthetic_data()


# Generate train dataset
# synthetic_dataset = generate_train_synthetic_data()

# Print examples
# for data in synthetic_dataset:
#     print(data, "\n")
