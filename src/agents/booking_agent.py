from src.utils.helpers import reformat_text


# RAG-based Agent
class BookingAgent:
    def __init__(self, intent_model, intent_tok, slot_model, slot_tok, response_model, response_tok, retriever):
        self.intent_model = intent_model
        self.intent_tok = intent_tok
        self.slot_model = slot_model
        self.slot_tok = slot_tok
        self.response_model = response_model
        self.response_tok = response_tok
        self.retriever = retriever  # Function to retrieve relevant data

    def extract_intent(self, user_input):
        """Use the intent classification model to get the user's intent."""
        inputs = self.intent_tok(user_input, return_tensors="pt", padding=True, truncation=True)
        output = self.intent_model.generate(**inputs)
        return self.intent_tok.decode(output[0], skip_special_tokens=True)

    def extract_slots(self, user_input):
        """Use the slot extraction model to get slot values from the user input."""
        inputs = self.slot_tok(user_input, return_tensors="pt", padding=True, truncation=True)
        output = self.slot_model.generate(**inputs)
        slot_output = self.slot_tok.decode(output[0], skip_special_tokens=True)
        slots = {}
        for pair in slot_output.split(", "):
            if "=" in pair:
                key, value = pair.split("=", 1)
                slots[key.strip()] = value.strip()
        return slots

    def retrieve_information(self, intent, slots):
        """Retrieve relevant data using the extracted intent and slots."""

        action = intent.split("_", 1)[0]
        product_type = slots.get("product-type")  # Extract product type

        if not product_type:
            return {}  # Return empty if product type is missing

        if action == "find":
            return self.retriever.find_product(product_type, **slots)

        if action == "buy":
            return self.retriever.buy_product(product_type, **slots)

        return {}

        # return self.retriever
        # if intent == "find_train":
        #     return self.retriever.find_train(
        #         train_day=slots.get("train-day"),
        #         train_departure=slots.get("train-departure"),
        #         train_destination=slots.get("train-destination")
        #     )
        # if intent == "find_product":
        #     category = slots.get("product-type")
        #
        #     if category == "food":
        #         return self.retriever.find_food(
        #             food_type=slots.get("food-type"),
        #             restaurant_name=slots.get("restaurant-name")
        #         )
        #
        #     if category == "shirt":
        #         return self.retriever.find_shirt(
        #             features=slots.get("feature"),
        #             type=slots.get("type"),
        #             size=slots.get("size"),
        #         )

    def generate_response(self, user_input):
            """Main agent function to process user input and generate a response."""
            # Extract intent
            intent = self.extract_intent(user_input)

            # Extract slots
            slots = self.extract_slots(user_input)

            print(f'augmented_input 0: {intent}, {slots}')

            # Retrieve relevant data
            retrieved_data = self.retrieve_information(intent, slots)

            # Augment input with retrieved data
            retrieved_text = " ".join([f"{key}: {value}" for key, value in retrieved_data.items()])
            augmented_input = f"generate response: {user_input} Intent: {intent}. Slots: {', '.join([f'{k}={v}' for k, v in slots.items()])}. Retrieved: {retrieved_text}"

            augmented_input = reformat_text(augmented_input)

            inputs = self.response_tok(augmented_input, return_tensors="pt", padding=True, truncation=True)
            output = self.response_model.generate(**inputs, max_length=128, num_beams=5, no_repeat_ngram_size=3)

            # Generate final response
            return self.response_tok.decode(output[0], skip_special_tokens=True)
