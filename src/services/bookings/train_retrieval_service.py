class TrainRetrievalService:
    """Service class responsible for retrieving different types of products."""

    @staticmethod
    def find_train(train_day, train_departure, train_destination):
        return {
            "train-day": train_day,
            "train-departure": train_departure,
            "train-destination": train_destination,
            "available_trains": 12
        }

    @staticmethod
    def buy_train(train_id, passenger_name):
        return {
            "train_id": train_id,
            "passenger_name": passenger_name,
            "status": "booked"
        }