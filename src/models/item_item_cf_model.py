import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.repository.mongodb_service import DatabaseService
from src.utils.asset_paths import AssetPaths
from src.utils.helpers import get_path_to
from src.utils.mongo_collections import MongoCollection


# Load the dataset
def load_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

# Convert product attributes into a text-based representation
def preprocess_data(products):
    processed_data = []
    product_ids = []

    for product in products:
        product_ids.append(product["product_id"])

        print(product['features'])

        # Combine text-based features into one string
        text_features = f"{product['brand']} {product['category']} {' '.join(product['features'])} " \
                        f"{' '.join(map(str, product['size']))} price:{product['price']} rating:{product['rating']}"

        processed_data.append(text_features)

    return product_ids, processed_data

# Compute similarity matrix
def compute_similarity(product_ids, text_features):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_features)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return product_ids, similarity_matrix

# Store similarity matrix in MongoDB
def store_similarity_in_mongo(product_ids, similarity_matrix):
    db_service = DatabaseService()

    db_service.delete_many(MongoCollection.ITEM_SIMILARITY.value, {})  # Clear previous data

    for idx, product_id in enumerate(product_ids):
        similar_items = {
            product_ids[j]: float(similarity_matrix[idx][j])
            for j in range(len(product_ids)) if j != idx
        }

        # Store in MongoDB
        db_service.insert_one(MongoCollection.ITEM_SIMILARITY.value, {
            "product_id": product_id,
            "similar_products": similar_items
        })

# Full pipeline execution
def train_similarity_model(json_path):
    print("Loading data...")
    products = load_data(json_path)

    print("Preprocessing data...")
    product_ids, text_features = preprocess_data(products)

    print("Computing similarity matrix...")
    product_ids, similarity_matrix = compute_similarity(product_ids, text_features)

    print("Storing similarity matrix in MongoDB...")
    store_similarity_in_mongo(product_ids, similarity_matrix)

    print("Training complete! Similarity matrix stored.")


def get_similar_products(product_id, top_n=5):
    result = DatabaseService().find_one(MongoCollection.ITEM_SIMILARITY.value, {"product_id": product_id})
    if result:
        sorted_similar = sorted(result["similar_products"].items(), key=lambda x: x[1], reverse=True)
        return sorted_similar[:top_n]
    return []

if __name__ == "__main__":
    # train_similarity_model(get_path_to(AssetPaths.ECOM_DATASET.value))

    # Example usage
    similar_products = get_similar_products("P0001")
    print(similar_products)
