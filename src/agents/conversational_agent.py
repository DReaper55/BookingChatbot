from src.agents.booking_agent import BookingAgent
from src.evaluation.evaluate_context_translation_model import ContextTranslator
from src.evaluation.evaluate_slot_filler_model import SlotFiller
from src.services.session_service import SessionService, ConversationSpeaker
from src.services.state_service import StateService, ConversationStates
from src.utils.singleton_meta import SingletonMeta


class ConversationalAgent(metaclass=SingletonMeta):
    """Handles the conversation flow using StateService, SlotFiller, and the Booking Agent."""
    def __init__(self):
        self.__slot_filler = SlotFiller()
        self.__booking_agent = BookingAgent()
        self.__context_translator = ContextTranslator()
        self.__session_service = SessionService()
        self.__state_service = StateService()

    def _is_greeting(self, user_message):
        """Detects if the user is greeting the bot."""
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        return user_message.lower() in greetings

    def handle_user_message(self, user_id, user_message):
        """Processes user input and determines whether to request more slots or proceed with booking."""
        print(f"State 0: {self.__state_service.get_state(user_id)}")

        self.__session_service.add_turn(user_id, ConversationSpeaker.USER.value, user_message)

        # Start of a conversation
        if self.__state_service.get_state(user_id) == ConversationStates.IDLE.value:
            if self._is_greeting(user_message):
                response = "Hi! How may I assist you today?"
                self.__session_service.add_turn(user_id, ConversationSpeaker.BOT.value, response)
                return response
            else:
                self.__state_service.update_state(user_id, ConversationStates.SLOT_FILLING.value)

        print(f"State 1: {self.__state_service.get_state(user_id)}")

        print(self.__session_service.get_context(user_id))

        if self.__state_service.get_state(user_id) == ConversationStates.SLOT_FILLING.value:
            response = self.__slot_filler.generate_response(self.__session_service.get_context(user_id))
            self.__session_service.add_turn(user_id, ConversationSpeaker.BOT.value, response)

            # If response contains a structured task, switch to booking
            if "." in response:
                self.__state_service.update_state(user_id, ConversationStates.CONTEXT_TRANSLATOR.value)
            else:
                return response

        if self.__state_service.get_state(user_id) == ConversationStates.CONTEXT_TRANSLATOR.value:
            task = self.__context_translator.generate_response(self.__session_service.get_context(user_id))
            user_message = task
            self.__state_service.update_state(user_id, ConversationStates.BOOKING.value)

        print(f"Task: {user_message}")

        if self.__state_service.get_state(user_id) == ConversationStates.BOOKING.value:
            # Get the first sentence of the last conversation as the task
            structured_task = user_message.split(".")[0] + "."  # Extract first sentence

            # Pass the task to the booking agent
            booking_response = self.__booking_agent.generate_response(structured_task)
            self.__session_service.add_turn(user_id, ConversationSpeaker.BOT.value, booking_response)

            # Reset state and session after booking is complete
            self.__state_service.reset_state(user_id)
            self.__session_service.reset_session(user_id, structured_task)

            return booking_response + " What would you like me to do next?"

        return "I'm not sure how to proceed."