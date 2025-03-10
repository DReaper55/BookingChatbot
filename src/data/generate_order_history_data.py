import json
import random
import uuid
from collections import defaultdict

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.repository.mongodb_service import DatabaseService
from src.utils.asset_paths import AssetPaths
from src.utils.helpers import get_path_to
from src.utils.mongo_collections import MongoCollection


# Load product data
def load_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

# Generate random user IDs
def assign_user_ids(reviews):
    user_map = {}
    for review in reviews:
        user = review["user"]
        if user not in user_map:
            user_map[user] = {
                "user_id": str(uuid.uuid4()),
                "user": user,
            }
    return user_map

# Create order history from product reviews
def generate_order_history(products):
    user_orders = defaultdict(lambda: {"user_name": "", "orders": []})

    # Collect all reviews to assign user IDs
    all_reviews = [review for product in products for review in product["reviews"]]
    user_map = assign_user_ids(all_reviews)

    for product in products:
        product_id = product["product_id"]
        product_name = product["name"]
        price = product["price"]
        category = product["category"]

        for review in product.get("reviews", []):
            user = review["user"]
            user_info = user_map[user]
            user_id = user_info["user_id"]
            user_name = user_info["user"]
            rating = review["rating"]

            # Simulate a purchase quantity (random 1-5)
            quantity = random.randint(1, 5)

            # Generate order entry
            order = {
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "price": price,
                "quantity": quantity,
                "total_price": price * quantity,
                "rating_given": rating
            }

            user_orders[user_id]["user"] = user_name
            user_orders[user_id]["orders"].append(order)

    return user_orders

# Store order history in MongoDB
def store_orders_in_mongo(user_orders):
    db_service = DatabaseService()

    db_service.delete_many(MongoCollection.USER_ORDERS.value, {})  # Clear previous data

    for user_id, data in user_orders.items():
        db_service.insert_one(MongoCollection.USER_ORDERS.value, {
            "user_id": user_id,
            "user": data["user"],
            "orders": data["orders"]
        })

# Full pipeline execution
def create_order_history(json_path):
    print("Loading data...")
    products = load_data(json_path)

    print("Generating order history...")
    user_orders = generate_order_history(products)

    print("Storing orders in MongoDB...")
    store_orders_in_mongo(user_orders)

    print("Order history generation complete!")


def get_user_orders(user_id):
    result = DatabaseService().find_one(MongoCollection.USER_ORDERS.value, {"user_id": user_id})
    return result if result else None


if __name__ == "__main__":
    create_order_history(get_path_to(AssetPaths.ECOM_DATASET.value))

    # user_data = get_user_orders("ce81c9bd-fa1d-4ddc-a61d-6b9722977194")
    # print(user_data["user"])  # Sophia
    # print(user_data["orders"])  # List of past purchases
