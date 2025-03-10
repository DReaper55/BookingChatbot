import os

from pymongo import MongoClient
from typing import List, Dict, Any, Optional

from src.utils.env_keys import EnvKeys
from src.utils.mongo_collections import MongoCollection
from dotenv import load_dotenv

load_dotenv()


class DatabaseService:
    def __init__(self, db_url: str = os.getenv(EnvKeys.MONGO_CLUSTER.value), db_name: str = os.getenv(EnvKeys.MONGO_DB.value)):
        """Initialize MongoDB connection."""
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]

        # Define collections
        self.products = self.db[MongoCollection.PRODUCTS.value]
        self.user_orders = self.db[MongoCollection.USER_ORDERS.value]
        self.item_similarity = self.db[MongoCollection.ITEM_SIMILARITY.value]
        self.user_similarity = self.db[MongoCollection.USER_SIMILARITY.value]

    # ---------------------- CRUD Operations ----------------------

    def insert_one(self, collection_name: str, document: Dict[str, Any]) -> Any:
        """Insert a single document into the given collection."""
        collection = self.db[collection_name]
        return collection.insert_one(document).inserted_id

    def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[Any]:
        """Insert multiple documents into the given collection."""
        collection = self.db[collection_name]
        return collection.insert_many(documents).inserted_ids


    def find_one(self, collection_name: str, query: Dict[str, Any], projection: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
        """Find a single document based on a query with an optional projection."""
        collection = self.db[collection_name]
        return collection.find_one(query, projection)

    def find_many(self, collection_name: str, query: Dict[str, Any], projection: Optional[Dict[str, int]] = None, limit: int = 0) -> List[Dict[str, Any]]:
        """Find multiple documents based on a query."""
        collection = self.db[collection_name]
        cursor = collection.find(query, projection).limit(limit)
        return list(cursor)

    def update_one(self, collection_name: str, query: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        """Update a single document in a collection."""
        collection = self.db[collection_name]
        result = collection.update_one(query, {"$push": update_data})
        return result.modified_count

    def update_one_replace(self, collection_name: str, query: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        """Update a single document in a collection."""
        collection = self.db[collection_name]
        result = collection.update_one(query, {"$set": update_data})
        return result.modified_count

    def update_many(self, collection_name: str, query: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        """Update multiple documents in a collection."""
        collection = self.db[collection_name]
        result = collection.update_many(query, {"$push": update_data})
        return result.modified_count

    def delete_one(self, collection_name: str, query: Dict[str, Any]) -> int:
        """Delete a single document from a collection."""
        collection = self.db[collection_name]
        result = collection.delete_one(query)
        return result.deleted_count

    def delete_many(self, collection_name: str, query: Dict[str, Any]) -> int:
        """Delete multiple documents from a collection."""
        collection = self.db[collection_name]
        result = collection.delete_many(query)
        return result.deleted_count

    # ---------------------- Utility Functions ----------------------

    def count_documents(self, collection_name: str, query: Dict[str, Any] = {}) -> int:
        """Count the number of documents in a collection based on a query."""
        collection = self.db[collection_name]
        return collection.count_documents(query)

    def create_index(self, collection_name: str, field: str, unique: bool = False):
        """Create an index on a field."""
        collection = self.db[collection_name]
        collection.create_index(field, unique=unique)

    def drop_collection(self, collection_name: str):
        """Drop a collection."""
        self.db[collection_name].drop()

    def close_connection(self):
        """Close the MongoDB connection."""
        self.client.close()
