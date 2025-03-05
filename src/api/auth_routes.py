import os

from fastapi import APIRouter, HTTPException, Security
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional
from fastapi.security import OAuth2PasswordBearer

from dotenv import load_dotenv

from src.repository.mongodb_service import DatabaseService
from src.utils.env_keys import EnvKeys
from src.utils.mongo_collections import MongoCollection

load_dotenv()

# Constants
SECRET_KEY = os.getenv(EnvKeys.SECRET_KEY.value)
ALGORITHM = os.getenv(EnvKeys.HASHING_ALGO.value)
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv(EnvKeys.ACCESS_TOKEN_EXPIRATION.value))

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Router
router = APIRouter()
db_service = DatabaseService()

# Models
class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

def hash_password(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Signup Endpoint
@router.post("/signup", response_model=TokenResponse)
def signup(user: SignupRequest):
    """Signup a new user."""
    existing_user = db_service.find_one(MongoCollection.USERS.value, {"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Hash password
    hashed_password = hash_password(user.password)

    # Save user
    user_id = db_service.insert_one(MongoCollection.USERS.value, {
        "username": user.username,
        "email": user.email,
        "password": hashed_password
    })

    # Generate JWT token
    access_token = create_access_token(data={"sub": user.username, "id": user_id})
    return {"access_token": access_token, "token_type": "bearer"}

# Login Endpoint
@router.post("/login", response_model=TokenResponse)
def login(user: LoginRequest):
    """Login user and return JWT token."""
    db_user = db_service.find_one(MongoCollection.USERS.value, {"username": user.username})

    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Extract user id from db_user
    user_id = str(db_user.get("_id"))

    # Generate JWT token
    access_token = create_access_token(data={"sub": user.username, "id": user_id})
    return {"access_token": access_token, "token_type": "bearer"}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Security(oauth2_scheme)):
    """Validate JWT token and return current user."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
