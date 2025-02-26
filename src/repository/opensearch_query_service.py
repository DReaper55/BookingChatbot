from src.repository.mongodb_opensearch_connector import sync_mongo_to_opensearch

opensearch_client, index_name = sync_mongo_to_opensearch(is_sync=False)


def search_products(query):
    search_query = {
        "_source": {
            "excludes": ["features.pages"]
        },
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["name", "brand", "category", "feature.*"]
            }
        }
    }

    response = opensearch_client.search(index=index_name, body=search_query)

    results = []
    for hit in response["hits"]["hits"]:
        results.append(hit["_source"])

    return results


# Example: Searching for "Nike Shoes"
# results = search_products("Nike Shoes")
# print(results)
