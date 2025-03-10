from src.repository.opensearch_query_service import ProductsRetrievalService


class ShirtRetrievalService:
    """Service class responsible for retrieving different types of shirts."""

    @staticmethod
    def find_shirt(**kwargs):
        # Extract query parameters from kwargs
        product_type = "shirt"
        result = ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
        return result

    @staticmethod
    def buy_shirt(**kwargs):
        product_type = "shirt"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
