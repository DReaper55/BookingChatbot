from src.repository.opensearch_query_service import ProductsRetrievalService


class ToyRetrievalService:
    """Service class responsible for retrieving different types of toys."""
    @staticmethod
    def find_toy(**kwargs):
        product_type = "toy"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)

    @staticmethod
    def buy_toy(**kwargs):
        product_type = "toy"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
