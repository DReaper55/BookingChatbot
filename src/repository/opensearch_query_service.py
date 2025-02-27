from opensearchpy import OpenSearch, RequestsHttpConnection
from pymongo import MongoClient
import os

from dotenv import load_dotenv

from src.utils.env_keys import EnvKeys

load_dotenv()


class ProductsRetrievalService:
    def __init__(self,
                 mongo_uri=os.getenv(EnvKeys.MONGO_HOST.value),
                 mongo_db=os.getenv(EnvKeys.MONGO_DB.value),
                 mongo_collection=os.getenv(EnvKeys.MONGO_COLLECTION.value),
                 opensearch_host="localhost",
                 opensearch_port=os.getenv(EnvKeys.OPENSEARCH_PORT.value),
                 opensearch_index=os.getenv(EnvKeys.OPENSEARCH_INDEX_NAME.value),
                 opensearch_user=os.getenv(EnvKeys.OPENSEARCH_USERNAME.value),
                 opensearch_password=os.getenv(EnvKeys.OPENSEARCH_PASSWORD.value),
                 ):

        # MongoDB Connection
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[mongo_db]
        self.collection = self.db[mongo_collection]

        # OpenSearch Connection
        self.opensearch_client = OpenSearch(
            hosts=[{"host": opensearch_host, "port": opensearch_port}],
            http_auth=(opensearch_user, opensearch_password),
            use_ssl=True,
            http_compress = True,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30,
            connection_class=RequestsHttpConnection
        )

        self.index_name = opensearch_index
        self.index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 4
                }
            },
            # "mappings": {
            #     "properties": {
            #         "name": {"type": "text"},
            #         "features": {
            #             "properties": {
            #                 "size": {"type": "text"},
            #                 "material": {"type": "text"},
            #                 "pages": {"type": "float"},
            #             }
            #         }
            #     }
            # }
        }


        # Ensure OpenSearch index exists
        if not self.opensearch_client.indices.exists(index=self.index_name):
            self.opensearch_client.indices.create(index=self.index_name, body=self.index_body)

    def sync_mongo_to_opensearch(self):
        """Fetches data from MongoDB and indexes it into OpenSearch."""
        for product in self.collection.find():
            product_id = str(product["_id"])  # Convert ObjectId to string
            document = {
                "product_id": product_id,
                "name": product.get("name"),
                "brand": product.get("brand"),
                "category": product.get("category"),
                "features": product.get("features", []),
                "price": product.get("price"),
                "stock": product.get("stock"),
                "rating": product.get("rating"),
                "reviews": product.get("reviews", [])
            }

            # Index the document into OpenSearch
            self.opensearch_client.index(index=self.index_name, id=product_id, body=document)

        print("Sync Completed! MongoDB Data Indexed in OpenSearch")

    def search_products(self, query, filters=None, size=5):
        """
        Searches for products in OpenSearch.

        :param query: Search query (text-based)
        :param filters: Dictionary of filters to apply (e.g., {"brand": "Nike", "category": "shoes"})
        :param size: Number of results to return
        :return: List of matching products
        """
        must_clauses = []

        # Add full-text search
        if query:
            must_clauses.append({
                "multi_match": {
                    "query": query,
                    "fields": ["name^3", "brand^2", "category", "features"],
                    "type": "best_fields"
                }
            })

        if filters:
            for key, value in filters.items():
                must_clauses.append({"term": {key: value.lower()}})

        search_query = {
            "_source": {
                "excludes": ["features.pages"]
            },
            "query": {
                "bool": {
                    "must": must_clauses
                }
            },
            "size": size
        }

        try:
            response = self.opensearch_client.search(index=self.index_name, body=search_query)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            print(f"OpenSearch Query Error: {e}")
            return []

    def retrieve_formatted_result(self, product_type, **kwargs):
        """
        Generic method to find a product based on the product type and filters.
        """
        brand = kwargs.get("brand")
        item_name = kwargs.get("type")  # Specific product name or type (e.g., "polo" for shirts)
        size = kwargs.get("size")
        color = kwargs.get("color")
        features = kwargs.get("features", [])

        # Build filters dynamically
        filters = {
            "category": product_type,
            "brand": brand,
            "size": size,
            "color": color
        }

        # Remove None values from filters
        filters = {k: v for k, v in filters.items() if v}

        # If features exist, add them as a terms filter
        if features:
            filters["features"] = features

        # Perform search
        results = (self.search_products(item_name, filters=filters))

        if not results:
            return {"available": "NONE"}

        product = results[0]

        return {
            "available": str(product.get("stock", "NONE")),
            "price": f"${product.get('price', 'NONE')}",
            "size": product.get("size", "NONE"),
            "color": product.get("color", "NONE"),
            "brand": product.get("brand", "NONE"),
            "features": product.get("features", [])
        }


# Example Usage
# if __name__ == "__main__":
#     product_service = ProductsRetrievalService()
#
#     Sync MongoDB to OpenSearch
#     product_service.sync_mongo_to_opensearch()
#
#     Search for "Nike Shoes"
#     results = product_service.search_products("Nike Shoes", filters={"category": "shoe"})
#     print(results)
