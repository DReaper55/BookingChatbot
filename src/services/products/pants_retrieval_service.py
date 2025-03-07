from src.repository.opensearch_query_service import ProductsRetrievalService


class PantsRetrievalService:
    """Service class responsible for retrieving different types of pantss."""
    @staticmethod
    def find_pants(**kwargs):
        product_type = "pants"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)

    @staticmethod
    def buy_pants(**kwargs):
        product_type = "pants"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
