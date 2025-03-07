from src.repository.opensearch_query_service import ProductsRetrievalService


class JacketRetrievalService:
    """Service class responsible for retrieving different types of jackets."""
    @staticmethod
    def find_jacket(**kwargs):
        product_type = "jacket"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)

    @staticmethod
    def buy_jacket(**kwargs):
        product_type = "jacket"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
