from pymongo import MongoClient
from opensearchpy import OpenSearch, RequestsHttpConnection
import os

from src.utils.env_keys import EnvKeys
from dotenv import load_dotenv
load_dotenv()

# Connect to MongoDB
mongo_client = MongoClient(os.getenv(EnvKeys.MONGO_HOST.value))
db = mongo_client[os.getenv(EnvKeys.MONGO_DB.value)]
collection = db[os.getenv(EnvKeys.MONGO_COLLECTION.value)]

http_auth = (os.getenv(EnvKeys.OPENSEARCH_USERNAME.value), os.getenv(EnvKeys.OPENSEARCH_PASSWORD.value))

# Connect to OpenSearch
opensearch_client = OpenSearch(
    hosts=[{"host": "localhost", "port": os.getenv(EnvKeys.OPENSEARCH_PORT.value)}],
    http_auth=http_auth,
    use_ssl=True,
    http_compress = True,
    verify_certs=False,
    ssl_show_warn=False,
    timeout=30,
    connection_class=RequestsHttpConnection
)

# Test connection
# print(opensearch_client.info())

# Define OpenSearch index name
index_name = os.getenv(EnvKeys.OPENSEARCH_INDEX_NAME.value)
index_body = {
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

# Remove index
# opensearch_client.indices.delete(index=index_name)

# Create the OpenSearch index if it doesn't exist
if not opensearch_client.indices.exists(index=index_name):
    opensearch_client.indices.create(index=index_name, body=index_body)
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

# Fetch data from MongoDB and index into OpenSearch
def sync_mongo_to_opensearch(is_sync=True):
    if not is_sync:
        return opensearch_client, index_name

    for product in collection.find():
        product_id = str(product["_id"])  # Convert ObjectId to string
        document = {
            "product_id": product_id,
            "name": product.get("name"),
            "brand": product.get("brand"),
            "category": product.get("category"),
            "features": product.get("features"),
            "price": product.get("price"),
            "stock": product.get("stock"),
            "rating": product.get("rating"),
            "reviews": product.get("reviews", [])
        }

        # Index the document into OpenSearch
        opensearch_client.index(index=index_name, id=product_id, body=document)

    print("Sync Completed! MongoDB Data Indexed in OpenSearch")

    return opensearch_client, index_name


# import schedule
# import time

# Schedule sync every 5 minutes
# schedule.every(5).minutes.do(sync_mongo_to_opensearch)
#
# while True:
#     schedule.run_pending()
#     time.sleep(1)


# Run the sync
# sync_mongo_to_opensearch()
