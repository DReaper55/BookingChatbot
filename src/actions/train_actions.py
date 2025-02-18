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
