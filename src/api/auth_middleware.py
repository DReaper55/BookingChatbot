import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import Request, HTTPException
from passlib.exc import InvalidTokenError
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt, JWTError, ExpiredSignatureError

from dotenv import load_dotenv

from src.utils.env_keys import EnvKeys

load_dotenv()

# Secret key and algorithm (same as in auth_routes.py)
SECRET_KEY = os.getenv(EnvKeys.SECRET_KEY.value)
ALGORITHM = os.getenv(EnvKeys.HASHING_ALGO.value)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate JWT token and attach user to request state."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/auth"):  # Skip auth routes
            return await call_next(request)

        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization token")

        token = auth_header.split(" ")[1]

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            request.state.user = payload.get("sub")  # Attach user to request
        except ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            print(f"Unexpected auth error: {e}")  # Debugging
            raise HTTPException(status_code=500, detail="Authentication failed")

        return await call_next(request)


def auth_middleware_factory():
    return AuthMiddleware
