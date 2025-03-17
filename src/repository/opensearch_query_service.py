from opensearchpy import OpenSearch, RequestsHttpConnection
from pymongo import MongoClient

from dotenv import load_dotenv

import sys
import os

from pymongo.errors import PyMongoError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.recommendation_service import get_cf_recommendations
from src.utils.env_keys import EnvKeys
from src.utils.mongo_collections import MongoCollection

load_dotenv()


class ProductsRetrievalService:
    def __init__(self,
                 mongo_uri=os.getenv(EnvKeys.MONGO_CLUSTER.value),
                 mongo_db=os.getenv(EnvKeys.MONGO_DB.value),
                 mongo_collection=MongoCollection.PRODUCTS.value,

                 # Set up opensearch credentials
                 # opensearch_host=os.getenv(EnvKeys.OPENSEARCH_HOST.value),
                 # opensearch_port=os.getenv(EnvKeys.OPENSEARCH_PORT.value),
                 # opensearch_index=os.getenv(EnvKeys.OPENSEARCH_INDEX_NAME.value),
                 # opensearch_user=os.getenv(EnvKeys.OPENSEARCH_USERNAME.value),
                 # opensearch_password=os.getenv(EnvKeys.OPENSEARCH_PASSWORD.value),
                 ):

        # MongoDB Connection
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[mongo_db]
        self.collection = self.db[mongo_collection]


    # ....................................
    # Use OpenSearch for indexing and query
    # Initial connection when app starts
    # ....................................
    # # OpenSearch Connection
        # self.opensearch_client = OpenSearch(
        #     hosts=[{"host": opensearch_host, "port": opensearch_port}],
        #     http_auth=(opensearch_user, opensearch_password),
        #     use_ssl=True,
        #     http_compress = True,
        #     verify_certs=False,
        #     ssl_show_warn=False,
        #     timeout=30,
        #     connection_class=RequestsHttpConnection
        # )
        #
        # self.index_name = opensearch_index
        # self.index_body = {
        #     'settings': {
        #         'index': {
        #             'number_of_shards': 4
        #         }
        #     },
        #     "mappings": {
        #         "properties": {
        #             "name": {"type": "text"},
        #             "brand": {"type": "text"},
        #             "category": {"type": "keyword"},
        #             "features": {"type": "keyword"},  # Treat features as keyword for filtering
        #             "size": {"type": "keyword"},  # Ensure size is indexed correctly
        #             "price": {"type": "float"},
        #             "stock": {"type": "integer"},
        #             "rating": {"type": "float"},
        #             "reviews": {"type": "nested"}
        #         }
        #     }
        # }
        #
        # # self.opensearch_client.indices.delete(index=self.index_name)
        #
        # # Ensure OpenSearch index exists
        # if not self.opensearch_client.indices.exists(index=self.index_name):
        #     self.opensearch_client.indices.create(index=self.index_name, body=self.index_body)

    # ....................................
    # Sync mongo to opensearch
    # ....................................
    # def sync_mongo_to_opensearch(self):
    #     """Fetches data from MongoDB and indexes it into OpenSearch."""
    #     for product in self.collection.find():
    #         product_id = str(product["_id"])  # Convert ObjectId to string
    #         document = {
    #             "id": product_id,
    #             "product_id": product.get("product_id"),
    #             "name": product.get("name"),
    #             "brand": product.get("brand"),
    #             "category": product.get("category"),
    #             "features": product.get("features", []),
    #             "size": product.get("size", []),
    #             "price": product.get("price"),
    #             "stock": product.get("stock"),
    #             "rating": product.get("rating"),
    #             "reviews": product.get("reviews", [])
    #         }
    #
    #         # Index the document into OpenSearch
    #         self.opensearch_client.index(index=self.index_name, id=product_id, body=document)
    #
    #     print("Sync Completed! MongoDB Data Indexed in OpenSearch")

    # ....................................
    # Query opensearch for a product
    # ....................................
    # def search_products(self, query, filters=None, size=5):
    #     """
    #     Searches for products in OpenSearch.
    #
    #     :param query: Search query (text-based)
    #     :param filters: Dictionary of filters to apply (e.g., {"brand": "Nike", "category": "shoes"})
    #     :param size: Number of results to return
    #     :return: List of matching products
    #     """
    #     must_clauses = []
    #
    #     # Add full-text search
    #     if query:
    #         must_clauses.append({
    #             "multi_match": {
    #                 "query": query,
    #                 "fields": ["category", "brand"],
    #                 "type": "best_fields",
    #                 "fuzziness": "AUTO"
    #             }
    #         })
    #
    #     # print(f'Filters: {filters}')
    #
    #     should_clauses = []
    #
    #     if filters:
    #         for key, value in filters.items():
    #             if key == "features":
    #                 for feature in value:
    #                     should_clauses.append({
    #                         "match": {
    #                             "features": {
    #                                 "query": feature,
    #                                 "fuzziness": "AUTO"
    #                             }
    #                         }
    #                     })
    #             elif key == "size":
    #                 should_clauses.append({
    #                     "match": {
    #                         "size": {
    #                             "query": value,
    #                             "fuzziness": "AUTO"
    #                         }
    #                     }
    #                 })
    #             else:
    #                 must_clauses.append({
    #                     "match": {
    #                         key: {
    #                             "query": value,
    #                             "fuzziness": "AUTO"
    #                         }
    #                     }
    #                 })
    #
    #     search_query = {
    #         "_source": {
    #             "excludes": ["features.pages"]
    #         },
    #         "query": {
    #             "bool": {
    #                 "must": must_clauses,
    #                 "should": should_clauses,
    #                 "minimum_should_match": 1  # At least one should match
    #             }
    #         },
    #         "size": 20
    #     }
    #
    #     try:
    #         response = self.opensearch_client.search(index=self.index_name, body=search_query)
    #         return [hit["_source"] for hit in response["hits"]["hits"]]
    #     except Exception as e:
    #         print(f"OpenSearch Query Error: {e}")
    #         return []

    # ....................................
    # Search Mongodb for product using
    # Mongo search index
    # ....................................
    def search_products(self, query, filters=None, size=10):
        import logging

        # MongoDB Atlas Search pipeline
        search_pipeline = []

        # Full-text search on category, brand, and features
        if query:
            search_pipeline.append({
                "$search": {
                    "index": "product_search",  # Your search index name
                    "text": {
                        "query": query,
                        "path": ["category", "brand", "features", "size"],  # Search across multiple fields
                        "fuzzy": {"maxEdits": 1}  # Allow minor typos
                    }
                }
            })

        # Apply filters
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                if key in ["features", "size"]:  # Arrays (e.g., ["cotton", "waterproof"])
                    if not isinstance(value, list):
                        value = [value]  # Ensure it's a list
                    filter_conditions.append({key: {"$elemMatch": {"$in": value}}})  # Match at least one element
                else:  # Partial text matching for other fields
                    filter_conditions.append({key: {"$regex": value, "$options": "i"}})

            if filter_conditions:
                search_pipeline.append({"$match": {"$and": filter_conditions}})

        # Limit results
        search_pipeline.append({"$limit": size})

        print(search_pipeline)

        try:
            # Execute aggregation query
            results = list(self.collection.aggregate(search_pipeline))
            return results
        except Exception as e:
            logging.error(f"MongoDB Query Error: {e}")
            return []


    def retrieve_formatted_result(self, product_type, **kwargs):
        """
        Generic method to find a product based on the product type and filters.
        """
        author = kwargs.get("author", None)
        price = kwargs.get("price", None)
        quantity = kwargs.get("quantity", 1)
        title = kwargs.get("title", None)
        brand = kwargs.get("brand", None)
        size = kwargs.get("size", None)
        category = kwargs.get("category", product_type)

        # List to store the values of keys that contain the word 'feature'
        features = [value for key, value in kwargs.items() if 'feature' in key]

        # Build filters dynamically
        filters = {
            "category": product_type,
            "author": author,
            "price": price,
            "quantity": quantity,
            "title": title,
            "brand": brand,
            "size": size,
        }

        # Remove None and 'null' values from filters
        filters = {k: v for k, v in filters.items() if v and v != 'null'}

        # If features exist, add them as a terms filter
        if features:
            filters["features"] = features

        print(f'augmented_input 3: {category}, {filters}')

        # Perform search
        results = (self.search_products(category, filters=filters))

        print(f"Search result: {results}")

        if not results:
            return {"available": "NONE"}

        filtered_products = get_best_product(results)[:10]

        # Use collaborative filtering to get top 5 recommendations
        # recommended_products = get_cf_recommendations(get_user_id(), filtered_products)
        # print(recommended_products)

        product = filtered_products[0]

        return {
            "available": str(product.get("stock", "NONE")),
            "price": f"${product.get('price', 'NONE')}",
            "size": product.get("size", "NONE"),
            "brand": product.get("brand", "NONE"),
            "id": product.get("product_id", "NONE"),
            "features": product.get("features", [])
        }


def get_user_id():
    return "0123"

def get_best_product(products):
    """
    Rank products based on multiple factors:
    - Matching score
    - Stock availability
    - Product rating
    - Review count
    """
    return sorted(
        products,
        key=lambda p: (
            p.get("stock", 0) > 0,  # Prioritize available products
            p.get("rating", 0),  # Higher ratings are better
            len(p.get("reviews", [])),  # More reviews = better trust
        ),
        reverse=True
    )

# Example Usage
# if __name__ == "__main__":
#     product_service = ProductsRetrievalService()
#
# #     Sync MongoDB to OpenSearch
# #     product_service.sync_mongo_to_opensearch()
#
#     filter = {'author': 'null', 'brand': 'Wrangler', 'feature-blue': 'blue', 'feature-denim': 'denim', 'feature-regular': 'regular', 'price': 'null', 'product-type': 'null', 'quantity': 'null', 'size': 'M', 'title': 'null', 'category': 'pants'}
#
# #     Search for "Nike Shoes"
# #     results = product_service.search_products("Adidas Shirt", filter)
#     results = product_service.retrieve_formatted_result("pants", filter=filter)
#     print(results)
