class AccessoryRetrievalService:
    """Service class responsible for retrieving different types of accessory."""
    @staticmethod
    def find_accessory(**kwargs):
        return {
            "available": "5",
            "price": "$25.25",
            "size": "XL"
        }

    @staticmethod
    def buy_accessory(**kwargs):
        return {
            "available": "5",
            "price": "$25.25",
            "size": "XL"
        }
