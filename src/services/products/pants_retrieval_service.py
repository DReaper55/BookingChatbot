class PantsRetrievalService:
    """Service class responsible for retrieving different types of pantss."""
    @staticmethod
    def find_pants(**kwargs):
        return {
            "available": "5",
            "price": "$25.25",
            "size": "XL"
        }

    @staticmethod
    def buy_pants(**kwargs):
        return {
            "available": "5",
            "price": "$25.25",
            "size": "XL"
        }
