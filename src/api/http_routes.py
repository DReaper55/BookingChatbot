from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.agents.conversational_agent import ConversationalAgent
from src.api.auth_routes import get_current_user
from src.repository.mongodb_service import DatabaseService
from src.utils.mongo_collections import MongoCollection

router = APIRouter()
agent = ConversationalAgent()  # Keep a running instance

class UserInput(BaseModel):
    user_id: str
    message: str

@router.post("/chat")
def chat(user: UserInput):
    response = agent.handle_user_message(user.user_id, user.message)
    return {"response": response}

@router.get("/")
def home():
    return {"message": "Conversational Agent is running!"}


db_service = DatabaseService()




# .......................................
# Products Collection CRUD
# .......................................

@router.post("/products", response_model=Dict[str, Any])
def create_product(product: Dict[str, Any]):
    """Create a new product."""
    product_id = db_service.insert_one(MongoCollection.PRODUCTS.value, product)
    return {"message": "Product created successfully", "product_id": str(product_id)}


@router.get("/products/{product_id}", response_model=Dict[str, Any])
def get_product(product_id: str):
    print(product_id)
    """Get a product by ID."""
    product = db_service.find_one(MongoCollection.PRODUCTS.value, {"product_id": product_id}, {"_id": 0})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.get("/products", response_model=List[Dict[str, Any]])
def get_all_product():
    """Get all products"""
    products = db_service.find_many(MongoCollection.PRODUCTS.value, {}, {"_id": 0})
    if not products:
        raise HTTPException(status_code=404, detail="Product not found")
    return products


@router.put("/products/{product_id}")
def update_product(product_id: str, update_data: Dict[str, Any]):
    """Update an existing product."""
    updated = db_service.update_one(MongoCollection.PRODUCTS.value, {"product_id": product_id}, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Product not found or not updated")
    return {"message": "Product updated successfully"}


@router.delete("/products/{product_id}")
def delete_product(product_id: str):
    """Delete a product."""
    deleted = db_service.delete_one(MongoCollection.PRODUCTS.value, {"product_id": product_id})
    if not deleted:
        raise HTTPException(status_code=404, detail="Product not found or not deleted")
    return {"message": "Product deleted successfully"}



# .......................................
# User Orders Collection CRUD
# .......................................

@router.post("/user_orders", response_model=Dict[str, Any])
def create_user_order(order: Dict[str, Any]):
    """Create a new user order."""
    order_id = db_service.insert_one(MongoCollection.USER_ORDERS.value, order)
    return {"message": "Order created successfully", "order_id": str(order_id)}


@router.get("/user_orders/{user_id}", response_model=List[Dict[str, Any]])
def get_user_orders(user_id: str):
    """Get all orders for a specific user."""
    orders = db_service.find_one(MongoCollection.USER_ORDERS.value, {"user_id": user_id}, {"_id": 0})
    return orders


@router.put("/user_orders/{order_id}")
def update_user_order(order_id: str, update_data: Dict[str, Any]):
    """Update an existing user order."""
    updated = db_service.update_one(MongoCollection.USER_ORDERS.value, {"order_id": order_id}, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Order not found or not updated")
    return {"message": "Order updated successfully"}


@router.delete("/user_orders/{order_id}")
def delete_user_order(order_id: str):
    """Delete a user order."""
    deleted = db_service.delete_one(MongoCollection.USER_ORDERS.value, {"order_id": order_id})
    if not deleted:
        raise HTTPException(status_code=404, detail="Order not found or not deleted")
    return {"message": "Order deleted successfully"}



# .......................................
# User Similarity Collection CRUD
# .......................................

@router.post("/user_similarity", response_model=Dict[str, Any])
def create_user_similarity(user_similarity: Dict[str, Any]):
    """Create a new user similarity entry."""
    sim_id = db_service.insert_one(MongoCollection.USER_SIMILARITY.value, user_similarity)
    return {"message": "User similarity data created successfully", "id": str(sim_id)}


@router.get("/user_similarity/{user_id}", response_model=Dict[str, Any])
def get_user_similarity(user_id: str):
    """Get a user's similarity data."""
    similarity_data = db_service.find_one(MongoCollection.USER_SIMILARITY.value, {"user_id": user_id}, {"_id": 0})
    if not similarity_data:
        raise HTTPException(status_code=404, detail="User similarity data not found")
    return similarity_data


@router.put("/user_similarity/{user_id}")
def update_user_similarity(user_id: str, update_data: Dict[str, Any]):
    """Update user similarity data."""
    updated = db_service.update_one(MongoCollection.USER_SIMILARITY.value, {"user_id": user_id}, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="User similarity data not found or not updated")
    return {"message": "User similarity data updated successfully"}


@router.delete("/user_similarity/{user_id}")
def delete_user_similarity(user_id: str):
    """Delete a user's similarity data."""
    deleted = db_service.delete_one(MongoCollection.USER_SIMILARITY.value, {"user_id": user_id})
    if not deleted:
        raise HTTPException(status_code=404, detail="User similarity data not found or not deleted")
    return {"message": "User similarity data deleted successfully"}



# .......................................
# User Collection CRUD
# .......................................

@router.post("/users", response_model=Dict[str, Any])
def create_user(user: Dict[str, Any]):
    """Create a new user entry."""
    user_id = db_service.insert_one(MongoCollection.USERS.value, user)
    return {"message": "User data created successfully", "id": str(user_id)}


@router.get("/users/{user_id}", response_model=Dict[str, Any])
def get_user(user_id: str):
    """Get a user's data."""
    similarity_data = db_service.find_one(MongoCollection.USERS.value, {"user_id": user_id}, {"_id": 0})
    if not similarity_data:
        raise HTTPException(status_code=404, detail="User data not found")
    return similarity_data


@router.put("/users/{user_id}")
def update_user(user_id: str, update_data: Dict[str, Any]):
    """Update user data."""
    updated = db_service.update_one(MongoCollection.USERS.value, {"user_id": user_id}, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="User data not found or not updated")
    return {"message": "User data updated successfully"}


@router.delete("/users/{user_id}")
def delete_user(user_id: str):
    """Delete a user's data."""
    deleted = db_service.delete_one(MongoCollection.USERS.value, {"user_id": user_id})
    if not deleted:
        raise HTTPException(status_code=404, detail="User data not found or not deleted")
    return {"message": "User data deleted successfully"}
