from src.agents.booking_agent import BookingAgent

agent = BookingAgent()

# Example user input
user_input = "Buy a new pair of Nike shoes in size 42 for men."
response = agent.generate_response(user_input)
print(response)
