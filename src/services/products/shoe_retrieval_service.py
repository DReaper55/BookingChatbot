from src.repository.opensearch_query_service import ProductsRetrievalService


class ShoeRetrievalService:
    """Service class responsible for retrieving different types of shoes."""
    @staticmethod
    def find_shoe(**kwargs):
        product_type = "shoe"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)

    @staticmethod
    def buy_shoe(**kwargs):
        product_type = "shoe"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
