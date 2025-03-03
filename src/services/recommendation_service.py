from src.repository.mongodb_service import DatabaseService
from src.utils.mongo_collections import MongoCollection


# ..................................
# Get orders of similar users,
# get similar items to what user wants,
# recommend similar products that
# similar users have bought before
# ..................................

def get_cf_recommendations(user_id, product_list):
    db_service = DatabaseService()
    
    # Get user-based recommendations
    similar_users = db_service.find_one(MongoCollection.USER_SIMILARITY.value, {"user_id": user_id})

    user_based_recommendations = []
    if similar_users:
        similar_users = sorted(similar_users["similar_users"].items(), key=lambda x: x[1], reverse=True)
        for similar_user, score in similar_users[:2]:
            user_orders = db_service.find_one(MongoCollection.USER_ORDERS.value, {"user": similar_user}) # todo: Check using IDs not username
            if user_orders:
                user_based_recommendations.extend(user_orders["orders"])

    # Get item-based recommendations
    item_based_recommendations = []
    for product in product_list:
        similar_items = db_service.find_one(MongoCollection.ITEM_SIMILARITY.value, {"product_id": product["product_id"]})
        if similar_items:
            similar_items = sorted(similar_items["similar_products"].items(), key=lambda x: x[1], reverse=True)
            item_based_recommendations.extend(similar_items[:1])

    # Convert user_based_recommendations to a set of product IDs for fast lookup
    user_recommended_product_ids = {order["product_id"] for order in user_based_recommendations}

    # Filter item_based_recommendations to keep only those in user_recommended_product_ids
    recommended_products_ids = [product_id for product_id, _ in item_based_recommendations if product_id in user_recommended_product_ids]

    recommended_products = [fetch_product_by_id(product_id) for product_id in recommended_products_ids[:5]]

    return recommended_products


def fetch_product_by_id(product_id):
    product = DatabaseService().find_one(MongoCollection.PRODUCTS.value, {"product_id": product_id}, {"_id": 0})  # Exclude MongoDB _id
    return product if product else {}  # Return empty dict if not found
