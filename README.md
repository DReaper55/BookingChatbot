# **AI-Powered Booking and Conversation Agent with Context-Aware Responses**

## **Project Overview**
This project is an AI-driven **Booking and Conversation Agent** designed to process user queries related to product searches and bookings. The system leverages multiple fine-tuned **T5 transformer models** to classify user intent, extract relevant features, generate structured tasks, and retrieve product data via a **Retrieval-Augmented Generation (RAG)** approach.

### **Key Features:**
- **Intent Classification** – Identifies the purpose of user queries (e.g., `find_product`, `buy_product`).
- **Slot Extraction** – Extracts relevant product attributes (size, color, brand, etc.).
- **Context Translator** – Converts free-text conversations into structured tasks.
- **Booking Model** – Generates personalized product recommendations based on retrieved data.
- **Retriever Service** – Queries a database (MongoDB or OpenSearch) to fetch matching products.
- **Conversation Agent** – Orchestrates dialogue flow, ensuring a seamless user experience.
- **WebSocket Support** – Enables real-time interactions.
- **Knowledge Distillation** – Optimizes models for reduced latency and improved efficiency.

---

## **Index**
1. [**Training the Booking Model**](#training-the-booking-model)
2. [**Dataset Format**](#dataset-format)
3. [**Generating Responses with the Booking Model**](#generating-responses-with-the-booking-model)
4. [**Context Translation Model**](#context-translation-model)
5. [**Intent Classifier Model**](#intent-classifier-model)
6. [**Feature Extraction Model**](#feature-extraction-model)
7. [**Slot Filler Model**](#slot-filler-model)
8. [**Booking Agent**](#booking-agent)
9. [**Retriever Service**](#retriever-service)
10. [**Conversation Agent**](#conversation-agent)
11. [**WebSocket API for Real-Time Communication**](#websocket-api-for-real-time-communication)
12. [**Knowledge Distillation for Model Compression**](#knowledge-distillation-for-model-compression)

---

## **Web App for Testing**
👉 **[Test the Web App Here](https://chatbot-gyiiekzy6-daniel-uwadis-projects.vercel.app)**

---

## **Resources**
- **Hugging Face Model Repository:** [Here](https://huggingface.co/DReaper)
- **Dataset Repository:** [Here](https://huggingface.co/DReaper)

---

# Booking Model

The **Booking Model** is responsible for generating responses based on product-related queries. It follows a **Retrieval-Augmented Generation (RAG)** approach, where it augments its response with actual product data retrieved from a database. This allows the model to provide more accurate and contextually relevant responses to users.

## **Training the Booking Model**

The model is fine-tuned on a dataset where each input consists of a user query, identified intent, extracted slot values, and retrieved product details. The training process involves:

- Loading a pre-trained **T5 model** and tokenizer.
- Using a structured dataset where responses are conditioned on retrieved data.
- Fine-tuning the model for **5 epochs** with a batch size of 8.

### **Training Script**
The following script loads and fine-tunes the T5 model:

```python
def train_model(output_dir):
    # Load model
    model, tokenizer, data_collator = load_t5_model_and_tokenizer(True, AssetPaths.T5_DISTIL_BOOKING_MODEL.value)

    # Load dataset
    train_dataloader, val_dataloader = load_booking_dataset()

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader,
        tokenizer=tokenizer,
    )

    # Train and save model
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
```

---

## **Dataset Format**

The dataset consists of structured inputs containing **user queries**, **intent classification**, **slot values**, and **retrieved product information**.

| Input Example | Output Example |
|--------------|---------------|
| `"generate response: I want to buy a navy blue hoodie in medium. Intent: buy_product. Slots: product-type=shirt, feature=navy blue, type=hoodie, size=M. Retrieved: feature=<FEATURE>, type=<TYPE>, size=<SIZE>, available=<AVAILABLE>, price=<PRICE>"` | `"The navy <FEATURE> <SIZE> <TYPE> is available for <PRICE>, and we have <AVAILABLE> in stock. Would you like to proceed with the purchase?"` |
| `"generate response: Do you have any long-sleeve shirts in size small? Intent: find_product. Slots: product-type=shirt, feature=long sleeves, size=S. Retrieved: feature=<FEATURE>, size=<SIZE>, available=<AVAILABLE>"` | `"Yes! We have <AVAILABLE> <FEATURE> shirts available in size <SIZE>. Would you like me to list them for you?"` |
| `"generate response: I'm looking for a yellow graphic t-shirt in large. Intent: find_product. Slots: product-type=shirt, feature=yellow, type=graphic t-shirt, size=L. Retrieved: feature=<FEATURE>, type=<TYPE>, size=<SIZE>, available=NONE"` | `"Unfortunately, we don't have any <FEATURE> <SIZE> graphic <TYPE> in stock at the moment. Would you like to explore other colors or styles?"` |

---

## **Generating Responses with the Booking Model**

Once the model is trained, it can be loaded and used to generate product-related responses based on **user queries, identified intent, slot values, and retrieved product data**.

### **Implementation**

```python
class BookingModel(metaclass=SingletonMeta):
    """Generates responses using a fine-tuned T5 model."""

    def __init__(self):
        self.__model, self.__tokenizer = None, None
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_response(self, user_input, active_intent="NONE", slot_values="NONE", retrieved="NONE"):
        """
        Generates a response based on user input, intent, slot values, and retrieved product data.
        """
        from src.utils.helpers import load_t5_model_and_tokenizer

        # Load trained model and tokenizer
        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, os.getenv(EnvKeys.RAG_BASED_BOOKING_MODEL.value))
        self.__model.to(self.__device)

        # Format input text for T5
        input_text = f"generate response: {user_input}. Intent: {active_intent}. Slots: {slot_values}. Retrieved: {retrieved}"
        input_text = reformat_text(input_text)

        # Tokenize input
        inputs = self.__tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=128, num_beams=5)

        # Decode output
        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
```

---

## **Example Usage**

You can use the **BookingModel** to generate responses based on user input:

```python
# Initialize the model
model = BookingModel()

# Define input details
user_input = "I'm looking for a denim jacket in size S"
active_intent = "find_product"
slot_values = "product-type=jacket, feature=denim, size=S"
retrieved = "feature=denim, available=20, price=$45, size=S"

# Generate response
response = model.generate_response(user_input, active_intent, slot_values, retrieved)
print("Bot:", response)
```

**Expected Output:**
```text
Bot: The denim jacket in size S is available for $45, and we have 20 in stock. Would you like to proceed with the purchase?
```
---

## Context Translation Model

The **Context Translation Model** is designed to convert a natural conversation into a structured task. It is based on a fine-tuned T5 model, which extracts the user's intent from a conversation and formulates a clear, actionable task.

### Training the Model

The model is trained using a dataset of human-bot conversations, where each conversation is mapped to a structured task. Below is the training process:

```python
def train_model(output_dir):
    # Load model
    model, tokenizer, data_collator = load_t5_model_and_tokenizer()

    # Load dataset
    train_dataloader, val_dataloader = load_context_translation_dataset()

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        learning_rate=3e-4,
        save_steps=500,
        save_total_limit=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
```

### Sample Training Data

The dataset consists of user-bot interactions paired with structured task outputs:

```json
[
  {
    "conversation": [
      "User: I feel like eating tacos tonight.",
      "Bot: There are 4 Mexican restaurants nearby. Would you like recommendations?",
      "User: Yes, that would be great."
    ],
    "structured_task": "Recommend 4 Mexican restaurants nearby."
  },
  {
    "conversation": [
      "User: Where can I get a good burger?",
      "Bot: Do you have a preferred restaurant or any price range in mind?",
      "User: Something affordable, but still good quality."
    ],
    "structured_task": "Find restaurants offering affordable but high-quality burgers near me."
  },
  {
    "conversation": [
      "User: I want to order pasta from Olive Garden.",
      "Bot: What type of pasta would you like?",
      "User: Spaghetti Carbonara."
    ],
    "structured_task": "Order Spaghetti Carbonara from Olive Garden."
  }
]
```

### Using the Trained Model

Once trained, the model can be used to generate structured tasks from conversations. The `ContextTranslator` class provides an interface to interact with the model:

```python
class ContextTranslator(metaclass=SingletonMeta):
    """Generates classified intent responses using a fine-tuned T5 model."""
    def __init__(self):
        self.__model, self.__tokenizer = None, None
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_response(self, user_input):
        from src.utils.helpers import load_t5_model_and_tokenizer

        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, os.getenv(EnvKeys.CONTEXT_TRANSLATOR_MODEL.value))
        self.__model.to(self.__device)

        if type(user_input) is not str:
            user_input = "\n".join(user_input)

        user_input = self.__format_input(user_input)

        user_input = f"translate context: {user_input}"
        inputs = self.__tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=100, num_beams=5)

        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return response
```

### Example Usage

```python
user_query = """
User: I'm looking for a shirt
Bot: Do you have a preferred brand?
User: Adidas
Bot: What color do you prefer?
User: Yellow
Bot: What material or style do you prefer?
User: In size XL
"""

response = ContextTranslator().generate_response(user_query)
print("Bot:", response)
```

#### Expected Output
```
Bot: Find a yellow Adidas XL shirt.
```

---

## Intent Classifier Model

The **Intent Classifier Model** is designed to classify user queries into predefined intent categories. This helps in understanding user requests and routing them to the appropriate response or action.

### Sample Training Data

The dataset consists of user queries paired with their corresponding intents:

```json
[
  {
    "input": "Show me alternatives to the Garmin Forerunner 245.",
    "output": "find_similar_products"
  },
  {
    "input": "What are the hottest fitness trackers this week?",
    "output": "get_trending_products"
  },
  {
    "input": "Buy 2 red winter jackets in size XL for $50 each.",
    "output": "buy_jacket"
  }
]
```

### Model Training

The **Intent Classifier Model** is built by fine-tuning a **T5 transformer model**. The training process involves mapping user queries to a structured intent label.

```python
def train_intent_classifier(output_dir):
    # Load pre-trained T5 model and tokenizer
    model, tokenizer, data_collator = load_t5_model_and_tokenizer()

    # Load intent classification dataset
    train_dataloader, val_dataloader = load_intent_classification_dataset()

    # Define training parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        learning_rate=3e-4,
        save_steps=500,
        save_total_limit=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
```

### Expected Behavior

The model learns to classify user input into structured intent labels. For example:

#### **Input:**
```
"What are some good alternatives to the Apple Watch?"
```
#### **Output:**
```
"find_similar_products"
```

---

## Feature Extraction Model

The **Feature Extraction Model** is a multitask model that extracts structured information (slots) and category labels from a given text. This allows for detailed attribute extraction, enabling better search, filtering, and product recommendations.

### Sample Training Data

The dataset consists of user queries labeled with extracted slots and corresponding categories:

```json
[
  {
    "input": "Find blue linen trousers in Medium, regular fit, Adidas, under $60.",
    "slots": {
      "price": "$60",
      "size": "Medium",
      "brand": "Adidas",
      "features": [
        "trousers",
        "medium",
        "slim fit",
        "blue",
        "linen",
        "adidas",
        "regular"
      ]
    },
    "category": "pants"
  },
  {
    "input": "Buy a metal handbag from Gucci for $180.",
    "slots": {
      "price": "$180",
      "brand": "Gucci",
      "features": [
        "gucci",
        "metal",
        "handbag"
      ]
    },
    "category": "accessory"
  }
]
```

### Interacting with the Trained Model

The `FeatureExtraction` class provides three key functionalities:
- **extract_slot**: Extracts structured attributes (e.g., price, size, brand).
- **retrieve_category**: Identifies the category of the product.
- **extract_features**: Extracts descriptive features from the input text.

```python
class FeatureExtraction(metaclass=SingletonMeta):
    """Extracts structured features, slots, and categories using a fine-tuned T5 model."""
    def __init__(self):
        self.__model, self.__tokenizer = None, None
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_slot(self, text):
        from src.utils.helpers import load_t5_model_and_tokenizer

        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, os.getenv(EnvKeys.FEATURE_EXTRACTION_MODEL.value))
        self.__model.to(self.__device)

        text = f"extract slot: {text}"
        inputs = self.__tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=100, num_beams=5)

        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

    def retrieve_category(self, text):
        from src.utils.helpers import load_t5_model_and_tokenizer

        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, os.getenv(EnvKeys.FEATURE_EXTRACTION_MODEL.value))
        self.__model.to(self.__device)

        text = f"retrieve category: {text}"
        inputs = self.__tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=50, num_beams=5)

        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

    def extract_features(self, text):
        from src.utils.helpers import load_t5_model_and_tokenizer

        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, os.getenv(EnvKeys.FEATURE_EXTRACTION_MODEL.value))
        self.__model.to(self.__device)

        text = f"extract features: {text}"
        inputs = self.__tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=128, num_beams=5)

        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
```

### Example Usage

```python
user_input = "Buy a yellow casual dress in extra-large size for $120 with PayPal and standard delivery."
result = FeatureExtraction().extract_features(user_input)
print(result)
```

#### Expected Output
```json
{
  "price": "$120",
  "size": "extra-large",
  "features": [
    "yellow",
    "casual",
    "dress"
  ]
}
```

---

## Slot Filler Model

The **Slot Filler Model** is designed to guide users in providing all necessary details for a product search. If any key product attributes (such as size, brand, or quantity) are missing, the model generates relevant follow-up questions to complete the request.

### Sample Training Data

The dataset consists of user queries and the corresponding follow-up questions generated to fill missing slots:

```json
[
  {
    "input": [
      "User: I want to buy a green tank top.",
      "Bot: What size do you need?",
      "User: XXL."
    ],
    "output": "How many would you like to purchase?"
  },
  {
    "input": [
      "User: I'm looking for a belt."
    ],
    "output": "Do you have a preferred brand?"
  }
]
```

### Interacting with the Trained Model

The `SlotFiller` class generates follow-up questions to extract necessary product details.

```python
class SlotFiller(metaclass=SingletonMeta):
    """Generates slot-filling responses using a fine-tuned T5 model."""
    def __init__(self):
        self.__model, self.__tokenizer = None, None
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_response(self, user_input):
        """Fills missing slots by generating a response."""
        from src.utils.helpers import load_t5_model_and_tokenizer

        self.__model, self.__tokenizer, _ = load_t5_model_and_tokenizer(True, os.getenv(EnvKeys.SLOT_FILLER_MODEL.value))
        self.__model.to(self.__device)

        user_input = f"ask question: {user_input}"
        inputs = self.__tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.__model.generate(**inputs, max_length=128, num_beams=5)

        response = self.__tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
```

### Example Usage

```python
user_input = "I want to buy a blue jacket."
response = SlotFiller().generate_response(user_input)
print("Bot:", response)
```

#### Expected Output
```text
Bot: What size do you need?
```

---

## Booking Agent

The **Booking Agent** is responsible for handling user requests related to product bookings. It interacts with multiple models and services, including:

- **Intent Classifier Model**: Determines the user's intent.
- **Feature Extraction Model**: Extracts relevant product attributes.
- **Booking Model**: Manages the booking process.
- **Retriever Service**: Queries MongoDB or OpenSearch to find relevant products.

### Booking Agent Workflow

1. **Classifies the User's Intent**
2. **Extracts Relevant Product Features**
3. **Retrieves Matching Products from the Database**
4. **Formats the Data and Generates a Response**

### Interacting with the Booking Agent

```python
def generate_response(self, user_input):
    """Main agent function to process user input and generate a response."""
    # Extract intent
    intent = self.__intent_model.generate_response(user_input)

    # Extract slots
    slots = self.__extract_slots(user_input)

    # Retrieve relevant data
    retrieved_data = self.__retrieve_information(intent, slots)

    slots = format_extracted_features(slots, True)
    retrieved_text = format_extracted_features(retrieved_data, True)

    return self.__response_model.generate_response(user_input, intent, slots, retrieved_text), retrieved_data
```

---

## Retriever Service

The **Retriever Service** is responsible for querying the database (MongoDB or OpenSearch) to find the most relevant products based on user requirements. It dynamically filters products using attributes like category, brand, size, price, and features.

### Retrieving a Product

```python
def retrieve_formatted_result(self, product_type, **kwargs):
    """
    Generic method to find a product based on the product type and filters.
    """
    author = kwargs.get("author", None)
    price = kwargs.get("price", None)
    quantity = kwargs.get("quantity", 1)
    title = kwargs.get("title", None)
    brand = kwargs.get("brand", None)
    size = kwargs.get("size", None)
    category = kwargs.get("category", product_type)

    # Extract feature-related filters
    features = [value for key, value in kwargs.items() if 'feature' in key]

    # Build filters dynamically
    filters = {
        "category": product_type,
        "author": author,
        "price": price,
        "quantity": quantity,
        "title": title,
        "brand": brand,
        "size": size,
    }

    # Remove None and 'null' values
    filters = {k: v for k, v in filters.items() if v and v != 'null'}

    # Include features if available
    if features:
        filters["features"] = features

    # Perform search
    results = self.search_products(category, filters=filters)

    if not results:
        return {"available": "NONE"}

    filtered_products = get_best_product(results)[:10]

    # Use collaborative filtering to get top 5 recommendations
    recommended_products = get_cf_recommendations(get_user_id(), filtered_products)

    product = filtered_products[0]

    return {
        "available": str(product.get("stock", "NONE")),
        "price": f"${product.get('price', 'NONE')}",
        "size": product.get("size", "NONE"),
        "brand": product.get("brand", "NONE"),
        "id": product.get("product_id", "NONE"),
        "features": product.get("features", [])
    }
```

### Example Usage

```python
user_input = "Find me a Nike running shoe under $100."
response, retrieved_data = BookingAgent().generate_response(user_input)
print("Bot:", response)
```

#### Expected Output

```text
Bot: We found a Nike running shoe in size M for $95. Would you like to place an order?
```

---

## Conversation Agent

The **Conversation Agent** is responsible for handling direct interactions with users. It orchestrates the conversation flow by determining which models and services to invoke based on user input.

### Responsibilities

- **Interacts with the Slot Filler model** to gather missing product details from the user.
- **Uses the Context Translator model** to convert user intent into structured tasks.
- **Communicates with the Booking Agent** to process user requests and retrieve relevant products.
- **Manages user sessions** via the Session Service to maintain conversation context.
- **Uses a State Controller** to determine which step in the conversation flow should be executed next.

### Conversation Flow

1. **Greeting & Initialization**
    - Identifies if the user is greeting the bot and responds accordingly.
    - If no greeting is detected, it moves to slot filling.

2. **Slot Filling**
    - Requests missing product details from the user.
    - Once sufficient information is gathered, it moves to context translation.

3. **Context Translation**
    - Converts the gathered information into a structured task for the Booking Agent.

4. **Booking & Product Retrieval**
    - Passes the task to the Booking Agent for processing.
    - Retrieves product details and presents the results to the user.

5. **Session Reset & Next Steps**
    - Resets conversation state after booking completion.
    - Asks the user how they’d like to proceed next.

### Interacting with the Conversation Agent

```python
def handle_user_message(self, user_id, chat_id: str, user_message):
    """Processes user input and determines whether to request more slots or proceed with booking."""

    self.__session_service.add_turn(user_id, chat_id, ConversationSpeaker.USER.value, user_message)

    # Start of a conversation
    if self.__state_service.get_state(user_id) == ConversationStates.IDLE.value:
        if self._is_greeting(user_message):
            response = "Hi! How may I assist you today?"
            self.__session_service.add_turn(user_id, chat_id, ConversationSpeaker.BOT.value, response)
            return {"message": response}
        else:
            self.__state_service.update_state(user_id, ConversationStates.SLOT_FILLING.value)

    # Keep requesting more information
    if self.__state_service.get_state(user_id) == ConversationStates.SLOT_FILLING.value:
        response = self.__slot_filler.generate_response(self.__session_service.get_context(user_id, chat_id))

        # If response contains a structured task, switch to booking
        if "." in response:
            self.__state_service.update_state(user_id, ConversationStates.CONTEXT_TRANSLATOR.value)
        else:
            self.__session_service.add_turn(user_id, chat_id, ConversationSpeaker.BOT.value, response)
            return {"message": response}

    # Summarize all received information and create a task
    if self.__state_service.get_state(user_id) == ConversationStates.CONTEXT_TRANSLATOR.value:
        task = self.__context_translator.generate_response(self.__session_service.get_context(user_id, chat_id))
        user_message = task
        self.__state_service.update_state(user_id, ConversationStates.BOOKING.value)

    # Perform booking task
    if self.__state_service.get_state(user_id) == ConversationStates.BOOKING.value:
        structured_task = user_message.split(".")[0] + "."  # Extract first sentence

        # Pass the task to the Booking Agent
        booking_response, retrieved_data = self.__booking_agent.generate_response(structured_task)

        message = booking_response + " What would you like me to do next?"
        products = {
            "action": "load_product",
            "data": retrieved_data
        }

        # Reset state and session after booking is complete
        self.__state_service.reset_state(user_id)

        return {
            "message": message,
            "products": products
        }

    return {"message": "I'm not sure how to proceed."}
```

### Example Usage

```python
user_id = "12345"
chat_id = "abcde"
user_message = "I want a leather wallet from Gucci under $200."

response = ConversationAgent().handle_user_message(user_id, chat_id, user_message)
print("Bot:", response["message"])
```

#### Expected Output

```text
Bot: Do you have a preferred color?
```

---

#### WebSocket API for Real-Time Communication

The API provides WebSocket support for **real-time communication**. Clients can connect to WebSockets for both **general interactions** and **chat-based conversations**.

##### Establishing a WebSocket Connection:
To establish a WebSocket connection, a client should connect to:
```plaintext
ws://localhost:8000/ws/{user_id}
```
Example using Python:
```python
import asyncio
import websockets
import json

async def websocket_test():
    user_id = "12345"
    async with websockets.connect(f"ws://localhost:8000/ws/{user_id}") as websocket:
        await websocket.send("Hello, WebSocket!")
        response = await websocket.recv()
        print(response)

asyncio.run(websocket_test())
```

##### Chat Interaction via WebSocket:
For **interactive chat**, the client should connect to:
```plaintext
ws://localhost:8000/ws/chat/{user_id}/{chat_id}
```
Example using Python:
```python
async def chat_session():
    user_id = "12345"
    chat_id = "67890"

    async with websockets.connect(f"ws://localhost:8000/ws/chat/{user_id}/{chat_id}") as websocket:
        user_message = "I want to buy a blue jacket."
        
        await websocket.send(user_message)  # Send user input
        response = await websocket.recv()   # Receive chatbot response
        
        print(f"Bot: {response}")

asyncio.run(chat_session())
```

---

## **Knowledge Distillation for Model Compression**
To optimize model performance while reducing computational overhead, we distilled five task-based models into smaller, efficient student models. This process reduced the model sizes from over **250MB** to **20-50MB**, making them more suitable for deployment without significant loss in accuracy.

### **1. Creating a Custom Student Model**
We define a **lightweight** T5-Tiny model by reducing the number of layers, attention heads, and hidden dimensions. This allows the student model to retain essential features of the teacher model while being significantly smaller.

```python
def get_student_model(teacher_tokenizer):
    # Define a compact T5 configuration
    tiny_config = T5Config(
        d_model=256,       
        d_ff=1024,          
        num_layers=4,      
        num_decoder_layers=4,  
        num_heads=6,       
        vocab_size=32128,  
        dropout_rate=0.1,  
        layer_norm_epsilon=1e-6,
        decoder_start_token_id=0,  
        pad_token_id=0,  
    )

    # Initialize the student model
    student_model = T5ForConditionalGeneration(tiny_config)
    student_tokenizer = teacher_tokenizer  # Using the same tokenizer as the teacher model

    return student_model, student_tokenizer
```

### **2. Training the Student Model**
We train the student model using **Knowledge Distillation (KD)**, where it learns both from labeled data and the teacher model’s soft outputs.

- **Cross-Entropy Loss**: Ensures the student model learns directly from ground-truth labels.
- **KL-Divergence Loss**: Helps the student model mimic the teacher model’s logits.

```python
def train_student():
    student_model, student_tokenizer = get_student_model(teacher_tokenizer)

    teacher_model.to(device)
    student_model.to(device)

    train_dataloader = load_data()

    optimizer = AdamW(student_model.parameters(), lr=5e-5)
    num_epochs = 2
    student_model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)  

            decoder_input_ids = teacher_model.prepare_decoder_input_ids_from_labels(labels=labels).to(device)

            # Forward pass
            outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )

            # Compute losses
            ce_loss = outputs.loss  
            student_logits = outputs.logits.view(-1, outputs.logits.shape[-1])
            teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])

            # Pad teacher logits if necessary
            teacher_logits_padded = F.pad(
                teacher_logits, 
                (0, 0, 0, student_logits.shape[0] - teacher_logits.shape[0]), 
                "constant", 
                -1e9  
            )

            kd_loss = torch.nn.functional.kl_div(
                torch.log_softmax(student_logits, dim=-1),  
                torch.softmax(teacher_logits_padded, dim=-1),  
                reduction="batchmean"
            )

            alpha = 0.5  # Balancing factor
            loss = ce_loss + alpha * kd_loss  

            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    # Save the trained student model
    student_model.save_pretrained(get_path_to(AssetPaths.T5_DISTIL_BOOKING_MODEL.value))
    student_tokenizer.save_pretrained(get_path_to(AssetPaths.T5_DISTIL_BOOKING_MODEL.value))
```

### **Results**
- Model size significantly reduced (**~250MB → 20-50MB**).
- Faster inference speeds with minimal accuracy degradation.
- Improved efficiency for deployment in resource-constrained environments.
