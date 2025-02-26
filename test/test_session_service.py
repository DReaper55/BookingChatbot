from src.services.session_service import SessionService

# Initialize session memory
memory = SessionService(window_size=4)

# Simulating a user conversation
user_id = "user_123"  # Unique identifier for each user

memory.add_turn(user_id, "User", "I want to buy shoes.")
memory.add_turn(user_id, "Bot", "What brand are you looking for?")
memory.add_turn(user_id, "User", "Nike.")
memory.add_turn(user_id, "Bot", "What size do you need?")
memory.add_turn(user_id, "User", "42.")

# Get conversation context
print(memory.get_context(user_id))
# Output: [
#   "Bot: What brand are you looking for?",
#   "User: Nike.",
#   "Bot: What size do you need?",
#   "User: 42."
# ]

# Define structured task and check if session should reset
structured_task = "Buy a new pair of Nike shoes in size 42 for men. Would you like some help?"

memory.reset_session(user_id, structured_task)

# Check if memory has been cleared
print(memory.get_context(user_id))  # Output: []
