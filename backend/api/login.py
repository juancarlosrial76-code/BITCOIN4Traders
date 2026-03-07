import os
import time
import threading
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext

router = APIRouter()
security = HTTPBearer()

_secret_key = os.getenv("SECRET_KEY", "")
if not _secret_key:
    import secrets as _secrets

    _secret_key = _secrets.token_hex(32)
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "SECRET_KEY not set in environment — using ephemeral random key. "
        "Set SECRET_KEY env var for persistent sessions."
    )
SECRET_KEY = _secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _build_users_db() -> dict:
    """Build user database from environment variables.

    Env vars:
        ADMIN_USERNAME  (default: admin)
        ADMIN_PASSWORD  (required — no default for security)
    """
    import logging as _log

    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "")
    if not password:
        import secrets as _sec

        password = _sec.token_urlsafe(16)
        _log.getLogger(__name__).warning(
            "ADMIN_PASSWORD not set — using random password '%s'. "
            "Set ADMIN_PASSWORD env var to use a fixed password.",
            password,
        )
    return {
        username: {
            "username": username,
            "hashed_password": pwd_context.hash(password),
            "role": "admin",
        }
    }


users_db = _build_users_db()


# ---------------------------------------------------------------------------
# Simple in-memory rate limiter: max 5 login attempts per IP per minute
# ---------------------------------------------------------------------------
_login_attempts: dict = {}  # {ip: [timestamp, ...]}
_login_lock = threading.Lock()
_MAX_ATTEMPTS = 5
_WINDOW_SECONDS = 60


def _check_rate_limit(client_ip: str) -> None:
    """Raise HTTP 429 if client exceeded login attempt limit."""
    now = time.time()
    with _login_lock:
        timestamps = _login_attempts.get(client_ip, [])
        # Remove timestamps outside the window
        timestamps = [t for t in timestamps if now - t < _WINDOW_SECONDS]
        if len(timestamps) >= _MAX_ATTEMPTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many login attempts. Try again in {_WINDOW_SECONDS}s.",
            )
        timestamps.append(now)
        _login_attempts[client_ip] = timestamps


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub") or ""
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = users_db.get(username)
    if user is None:
        raise credentials_exception
    return user


class LoginRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/login", response_model=Token)
async def login(request: LoginRequest, http_request: Request):
    client_ip = http_request.client.host if http_request.client else "unknown"
    _check_rate_limit(client_ip)
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"], "role": current_user["role"]}
