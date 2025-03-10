import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.retrieval_service import RetrievalService
from src.utils.singleton_meta import SingletonMeta


class Retriever(metaclass=SingletonMeta):
    def __init__(self):
        self.__retrieval_service = RetrievalService()

    def find_product(self, product_type, **kwargs):
        method_name = f"find_{product_type}"
        retrieval_method = getattr(self.__retrieval_service, method_name, None)

        if retrieval_method:
            return retrieval_method(**kwargs)  # Pass only valid arguments
        return {}

    def buy_product(self, product_type, **kwargs):
        method_name = f"buy_{product_type}"
        retrieval_method = getattr(self.__retrieval_service, method_name, None)

        if retrieval_method:
            return retrieval_method(**kwargs)  # Pass only valid arguments
        return {}
