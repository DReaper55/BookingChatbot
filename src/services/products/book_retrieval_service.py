from src.repository.opensearch_query_service import ProductsRetrievalService


class BookRetrievalService:
    """Service class responsible for retrieving different types of books."""
    @staticmethod
    def find_book(**kwargs):
        product_type = "book"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)

    @staticmethod
    def buy_book(**kwargs):
        product_type = "book"
        return ProductsRetrievalService().retrieve_formatted_result(product_type, **kwargs)
