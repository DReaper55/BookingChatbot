from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.singleton_meta import SingletonMeta


class ConversationStates(Enum):
    IDLE = "idle"
    SLOT_FILLING = "slot-filling"
    BOOKING = "booking"
    CONTEXT_TRANSLATOR = "context-translator"


class StateService(metaclass=SingletonMeta):
    """Manages conversation state (slot-filling or booking)."""
    def __init__(self):
        self.states = {}  # Keeps track of user states

    def get_state(self, user_id):
        """Returns the current state of the user."""
        return self.states.get(user_id, ConversationStates.IDLE.value)

    def update_state(self, user_id, state):
        """Updates the state of the user."""
        self.states[user_id] = state

    def reset_state(self, user_id):
        """Resets the state after task completion."""
        self.states[user_id] = ConversationStates.IDLE.value