from src.repository.opensearch_query_service import ProductsRetrievalService


class FoodRetrievalService:
    """Service class responsible for retrieving different types of food."""
    @staticmethod
    def find_food(**kwargs):
        product_type = "food"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)

    @staticmethod
    def buy_food(**kwargs):
        product_type = "food"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
