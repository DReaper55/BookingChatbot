from enum import Enum

class MongoCollection(Enum):
    PRODUCTS = 'products'
    USER_ORDERS = 'user_orders'
    ITEM_SIMILARITY = 'item_similarity'
    USER_SIMILARITY = 'user_similarity'
