from src.repository.opensearch_query_service import ProductsRetrievalService


class AccessoryRetrievalService:
    """Service class responsible for retrieving different types of accessory."""
    @staticmethod
    def find_accessory(**kwargs):
        product_type = "accessory"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)

    @staticmethod
    def buy_accessory(**kwargs):
        product_type = "accessory"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
