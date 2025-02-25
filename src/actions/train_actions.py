# Function to retrieve train details
class Retriever:
    @staticmethod
    def find_train(train_day, train_departure, train_destination):
        # Simulated database call
        train_data = {
            "train-day": train_day,
            "train-departure": train_departure,
            "train-destination": train_destination,
            "available_trains": 12  # Example response
        }
        return train_data

    @staticmethod
    def find_food(food_type, restaurant_name):
        # Simulated database call
        train_data = {
            "food-type": food_type,
            "restaurant-name": restaurant_name,
            "price": "12.25",
            "estimated-delivery": "5 mins"
        }
        return train_data

    @staticmethod
    def find_shirt(features, type, size):
        # Simulated database call
        train_data = {
            # "feature": "blue",
            # "type": "shirt",
            "available": "5",
            "price": "$25.25",
            "size": "XL"
        }
        return train_data
