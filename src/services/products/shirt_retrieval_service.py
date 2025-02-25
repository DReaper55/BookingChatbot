class ShirtRetrievalService:
    """Service class responsible for retrieving different types of shirts."""
    @staticmethod
    def find_shirt(**kwargs):
        # print(kwargs.get("product-type"))
        return {
            "available": "5",
            "price": "$25.25",
            "size": "XL"
        }

    @staticmethod
    def buy_shirt(**kwargs):
        return {
            "available": "5",
            "price": "$25.25",
            "size": "XL"
        }
