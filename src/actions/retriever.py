from src.services.retrieval_service import RetrievalService


class Retriever:
    def __init__(self):
        self.retrieval_service = RetrievalService()

    def find_product(self, product_type, **kwargs):
        method_name = f"find_{product_type}"
        retrieval_method = getattr(self.retrieval_service, method_name, None)

        if retrieval_method:
            return retrieval_method(**kwargs)  # Pass only valid arguments
        return {}

    def buy_product(self, product_type, **kwargs):
        method_name = f"buy_{product_type}"
        retrieval_method = getattr(self.retrieval_service, method_name, None)

        if retrieval_method:
            return retrieval_method(**kwargs)  # Pass only valid arguments
        return {}
