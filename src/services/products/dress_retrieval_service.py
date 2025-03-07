from src.repository.opensearch_query_service import ProductsRetrievalService


class DressRetrievalService:
    """Service class responsible for retrieving different types of dresss."""
    @staticmethod
    def find_dress(**kwargs):
        product_type = "dress"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)

    @staticmethod
    def buy_dress(**kwargs):
        product_type = "dress"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
