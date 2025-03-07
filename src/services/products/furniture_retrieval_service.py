from src.repository.opensearch_query_service import ProductsRetrievalService


class FurnitureRetrievalService:
    """Service class responsible for retrieving different types of furnitures."""
    @staticmethod
    def find_furniture(**kwargs):
        product_type = "furniture"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)

    @staticmethod
    def buy_furniture(**kwargs):
        product_type = "furniture"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
