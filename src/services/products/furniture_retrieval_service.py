class FurnitureRetrievalService:
    """Service class responsible for retrieving different types of furnitures."""
    @staticmethod
    def find_furniture(**kwargs):
        return {
            "available": "5",
            "price": "$25.25",
            "size": "XL"
        }

    @staticmethod
    def buy_furniture(**kwargs):
        return {
            "available": "5",
            "price": "$25.25",
            "size": "XL"
        }
