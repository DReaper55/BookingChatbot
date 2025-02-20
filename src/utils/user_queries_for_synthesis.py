def get_train_queries():
    # Sample retrieved information simulation
    retrieved_data_samples = [
        {"train-day": "monday", "train-departure": "norwich", "train-destination": "cambridge", "available_trains": 12},
        {"train-day": "friday", "train-departure": "london", "train-destination": "oxford", "available_trains": 8, "first_departure": "06:00", "last_departure": "22:00"},
        {"train-day": "sunday", "train-departure": "birmingham", "train-destination": "manchester", "available_trains": 5, "average_price": "£35"},
    ]

    # Sample user queries
    user_queries = [
        "I need a train from London to Oxford on Friday.",
        "Can you find me a train from Birmingham to Manchester on Sunday?",
        "I'd like to book a train from Norwich to Cambridge.",
        "What are the earliest and latest trains from London to Oxford?",
    ]

    return user_queries, retrieved_data_samples

def get_hotel_queries():
    # Sample retrieved information simulation
    retrieved_data_samples = [
        {"hotel-location": "london", "check-in": None, "check-out": None, "available-hotels": 45},
        {"hotel-location": "new york", "check-in": "march 5", "check-out": "march 10", "available-hotels": 32, "average-price": "$150/night", "top-rated-hotel": "Grand Central Hotel"},
        {"hotel-location": "paris", "hotel-preference": "near Eiffel Tower", "available-hotels": 18, "first-available-check-in": "March 8", "last-available-check-out": "March 15"},
    ]

    # Sample user queries
    user_queries = [
        "I need a hotel in London for three nights.",
        "I’d like a hotel in New York from March 5 to March 10.",
        "I need a hotel in Paris near the Eiffel Tower.",
        "Find me a hotel in Tokyo for next weekend.",
    ]

    return user_queries, retrieved_data_samples

