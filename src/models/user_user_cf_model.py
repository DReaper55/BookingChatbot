import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Extract user profiles from reviews
def create_user_profiles(products):
    user_profiles = defaultdict(lambda: {"products": [], "ratings": {}})

    for product in products:
        product_id = product["product_id"]
        category = product["category"]
        brand = product["brand"]
        features = " ".join(product["features"])

        for review in product.get("reviews", []):
            user = review["user"]
            rating = review["rating"]

            # Store user preferences
            user_profiles[user]["products"].append(f"{brand} {category} {features}")
            user_profiles[user]["ratings"][product_id] = rating

    return user_profiles

# Convert user profiles into TF-IDF vectors
def compute_user_similarity(user_profiles):
    user_ids = list(user_profiles.keys())
    user_texts = [" ".join(profile["products"]) for profile in user_profiles.values()]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(user_texts)

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return user_ids, similarity_matrix

# Store user similarity in MongoDB
def store_similarity_in_mongo(user_ids, similarity_matrix):
    db_service = DatabaseService()

    db_service.delete_many(MongoCollection.USER_SIMILARITY.value, {})  # Clear previous data

    for idx, user_id in enumerate(user_ids):
        similar_users = {
            user_ids[j]: float(similarity_matrix[idx][j])
            for j in range(len(user_ids)) if j != idx
        }

        # Store in MongoDB
        db_service.insert_one(MongoCollection.USER_SIMILARITY.value, {
            "user": user_id,
            "similar_users": similar_users
        })

# Full pipeline execution
def train_user_similarity(json_path):
    print("Loading data...")
    products = load_data(json_path)

    print("Extracting user profiles...")
    user_profiles = create_user_profiles(products)

    print("Computing user similarity matrix...")
    user_ids, similarity_matrix = compute_user_similarity(user_profiles)

    print("Storing similarity matrix in MongoDB...")
    store_similarity_in_mongo(user_ids, similarity_matrix)

    print("Training complete! User similarity matrix stored.")


def get_user_id_map():
    """
    Retrieve user_id and user mapping from user_orders collection.
    """
    db_service = DatabaseService()

    user_map = {}
    for record in db_service.find_many(MongoCollection.USER_ORDERS.value, {}, {"user_id": 1, "user": 1}):
        user_map[record["user"]] = record["user_id"]
    return user_map

def update_user_similarity():
    """
    Update user_similarity collection by replacing "user" with corresponding "user_id".
    """
    db_service = DatabaseService()

    user_map = get_user_id_map()

    for record in db_service.find_many(MongoCollection.USER_SIMILARITY.value, {}):
        user = record["user"]
        if user in user_map:
            user_id = user_map[user]

            # Update the record in MongoDB
            db_service.update_one_replace(
                MongoCollection.USER_SIMILARITY.value,
                {"_id": record["_id"]},
                {"user_id": user_id}
            )
            print(f"Updated {user} -> {user_id}")


def get_similar_users(user_id, top_n=5):
    result = DatabaseService().find_one(MongoCollection.USER_SIMILARITY.value, {"user": user_id})
    if result:
        sorted_similar = sorted(result["similar_users"].items(), key=lambda x: x[1], reverse=True)
        return sorted_similar[:top_n]
    return []


# if __name__ == "__main__":
#     # train_user_similarity(get_path_to(AssetPaths.ECOM_DATASET.value))
#
#     similar_users = get_similar_users("Sophia")
#     print(similar_users)
#
#     # update_user_similarity()
