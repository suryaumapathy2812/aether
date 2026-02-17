"""JWT token creation and validation."""

from __future__ import annotations

import os
import uuid
import hashlib
import time

import jwt

SECRET = os.environ.get("JWT_SECRET", "change_me_in_production")
ALGORITHM = "HS256"
TOKEN_EXPIRY = 60 * 60 * 24 * 365  # 1 year for device tokens


def hash_password(password: str) -> str:
    salt = uuid.uuid4().hex[:16]
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}:{hashed.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt, hashed = stored.split(":")
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return check.hex() == hashed


def create_token(user_id: str, device_id: str | None = None, expiry: int = TOKEN_EXPIRY) -> str:
    payload = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + expiry,
    }
    if device_id:
        payload["device"] = device_id
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    except jwt.InvalidTokenError:
        return None
