import re
from collections import deque
from enum import Enum
from typing import List

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.repository.mongodb_service import DatabaseService
from src.utils.helpers import get_path_to
from src.utils.mongo_collections import MongoCollection
from src.utils.singleton_meta import SingletonMeta


class ConversationSpeaker(Enum):
    USER = "User"
    BOT = "Bot"


# ....................................
# Handle Session using memory
# ....................................
# class SessionService(metaclass=SingletonMeta):
#     def __init__(self, window_size=4):
#         self.sessions = {}  # Stores session data per user
#         self.window_size = window_size
#
#     def _is_task_complete(self, structured_task):
#         """Checks if the structured task is complete (ends with a period or has multiple sentences)."""
#         return bool(re.search(r"\.\s|\.$", structured_task))
#
#     def add_turn(self, user_id, speaker, message):
#         """Adds a conversation turn to the session memory."""
#         if user_id not in self.sessions:
#             self.sessions[user_id] = deque(maxlen=self.window_size)
#
#         self.sessions[user_id].append(f"{speaker}: {message}")
#
#     def get_context(self, user_id):
#         """Returns the conversation history for a user within the window."""
#         return list(self.sessions.get(user_id, []))
#
#     def reset_session(self, user_id, structured_task):
#         """Resets the memory if the structured task is complete."""
#         if self._is_task_complete(structured_task):
#             self.sessions.pop(user_id, None)  # Remove session
#
#     def clear_all_sessions(self):
#         """Clears all stored session data."""
#         self.sessions.clear()


# ....................................
# Handle Session using local json file
# ....................................
# import json
# import os
#
#
# class SessionService(metaclass=SingletonMeta):
#     """Manages session storage using a JSON file for persistence."""
#     FILE_PATH = get_path_to('data/raw/session0.json')
#
#     def __init__(self):
#         self.sessions = {}
#         self.load_sessions()
#
#     def load_sessions(self):
#         """Loads existing session data from the JSON file."""
#         if os.path.exists(self.FILE_PATH):
#             with open(self.FILE_PATH, "r") as file:
#                 self.sessions = json.load(file)
#
#     def save_sessions(self):
#         """Saves the current session data to the JSON file."""
#         with open(self.FILE_PATH, "w") as file:
#             json.dump(self.sessions, file)
#
#     def add_turn(self, user_id, speaker, message):
#         """Stores a conversation turn in the JSON session."""
#         if user_id not in self.sessions:
#             self.sessions[user_id] = []
#         self.sessions[user_id].append(f"{speaker}: {message}")
#         self.save_sessions()
#
#     def get_context(self, user_id):
#         """Retrieves conversation history from the JSON file."""
#         return self.sessions.get(user_id, [])
#
#     def __is_task_complete(self, structured_task):
#         """Checks if the structured task is complete (ends with a period or has multiple sentences)."""
#         return bool(re.search(r"\.\s|\.$", structured_task))
#
#     def reset_session(self, user_id, structured_task):
#         """Clears the user's session and updates the file."""
#         if user_id in self.sessions and self.__is_task_complete(structured_task):
#             del self.sessions[user_id]
#             self.save_sessions()
#
#     def clear_all_sessions(self):
#         """Clears all stored session data."""
#         del self.sessions
#         self.save_sessions()


# ....................................
# Handle Session using Database
# ....................................
class SessionService(metaclass=SingletonMeta):
    """Manages session storage using MongoDB for persistence."""

    def __init__(self):
        self.db_service = DatabaseService()

    def add_turn(self, user_id: str, chat_id: str, speaker: str, message: str):
        """Stores a conversation turn in MongoDB."""
        session = self.db_service.find_one(MongoCollection.SESSIONS.value, {"user_id": user_id, "chat_id": chat_id})
        turn = f"{speaker}: {message}"

        if session:
            self.db_service.update_one(
                MongoCollection.SESSIONS.value,
                {"user_id": user_id, "chat_id": chat_id},
                {"conversation": turn}
            )
        else:
            self.db_service.insert_one(MongoCollection.SESSIONS.value, {"user_id": user_id, "chat_id": chat_id, "conversation": [turn]})

    def get_context(self, user_id: str, chat_id: str) -> List[str]:
        """Retrieves conversation history from MongoDB."""
        session = self.db_service.find_one(MongoCollection.SESSIONS.value, {"user_id": user_id, "chat_id": chat_id}, {"_id": 0, "conversation": 1})
        return session["conversation"] if session else []

    def __is_task_complete(self, structured_task: str) -> bool:
        """Checks if the structured task is complete (ends with a period or has multiple sentences)."""
        return bool(re.search(r"\.\s|\.$", structured_task))

    def reset_session(self, user_id: str, chat_id: str, structured_task: str):
        """Clears the user's session from MongoDB."""
        if self.__is_task_complete(structured_task):
            self.db_service.delete_one(MongoCollection.SESSIONS.value, {"user_id": user_id, "chat_id": chat_id})

    def clear_all_sessions(self):
        """Clears all session data from MongoDB."""
        self.db_service.delete_many(MongoCollection.SESSIONS.value, {})
