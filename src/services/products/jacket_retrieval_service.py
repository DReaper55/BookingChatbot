class JacketRetrievalService:
    """Service class responsible for retrieving different types of jackets."""
    @staticmethod
    def find_jacket(**kwargs):
        return {
            "available": "5",
            "price": "$25.25",
            "size": "XL"
        }

    @staticmethod
    def buy_jacket(**kwargs):
        return {
            "available": "5",
            "price": "$25.25",
            "size": "XL"
        }
