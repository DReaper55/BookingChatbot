class FoodRetrievalService:
    """Service class responsible for retrieving different types of food."""
    @staticmethod
    def find_food(food_type, restaurant_name):
        return {
            "food-type": food_type,
            "restaurant-name": restaurant_name,
            "price": "12.25",
            "estimated-delivery": "5 mins"
        }

    @staticmethod
    def buy_food(food_id, quantity):
        return {
            "food_id": food_id,
            "quantity": quantity,
            "status": "ordered"
        }
