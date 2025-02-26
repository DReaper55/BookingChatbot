from src.actions.retriever import Retriever
from src.evaluation.evaluate_booking_model import BookingModel
from src.evaluation.evaluate_intent_classifier_model import IntentClassifier
from src.evaluation.evaluate_slot_extraction_model import SlotExtraction
from src.utils.singleton_meta import SingletonMeta


# RAG-based Agent
class BookingAgent(metaclass=SingletonMeta):
    def __init__(self):
        self.__intent_model = IntentClassifier()
        self.__slot_model = SlotExtraction()
        self.__response_model = BookingModel()
        self.__retriever = Retriever()  # Tool to retrieve relevant data

    def __extract_slots(self, user_input):
        """Use the slot extraction model to get slot values from the user input."""
        slot_output = self.__slot_model.generate_response(user_input)
        slots = {}
        for pair in slot_output.split(", "):
            if "=" in pair:
                key, value = pair.split("=", 1)
                slots[key.strip()] = value.strip()
        return slots

    def __retrieve_information(self, intent, slots):
        """Retrieve relevant data using the extracted intent and slots."""

        action = intent.split("_", 1)[0]
        product_type = slots.get("product-type")  # Extract product type

        if not product_type:
            return {}  # Return empty if product type is missing

        if action == "find":
            return self.__retriever.find_product(product_type, **slots)

        if action == "buy":
            return self.__retriever.buy_product(product_type, **slots)

        return {}

    def generate_response(self, user_input):
            """Main agent function to process user input and generate a response."""
            # Extract intent
            intent = self.__intent_model.generate_response(user_input)

            # Extract slots
            slots = self.__extract_slots(user_input)

            print(f'augmented_input 0: {intent}, {slots}')

            # Retrieve relevant data
            retrieved_data = self.__retrieve_information(intent, slots)

            # Augment input with retrieved data
            retrieved_text = " ".join([f"{key}: {value}" for key, value in retrieved_data.items()])

            return self.__response_model.generate_response(user_input, intent, slots, retrieved_text)
