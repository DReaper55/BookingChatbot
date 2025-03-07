from src.repository.opensearch_query_service import ProductsRetrievalService


class ElectronicsRetrievalService:
    """Service class responsible for retrieving different types of electronics."""
    @staticmethod
    def find_electronics(**kwargs):
        product_type = "electronics"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)

    @staticmethod
    def buy_electronics(**kwargs):
        product_type = "electronics"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
