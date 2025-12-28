from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import jwt
import time
import logging
from datetime import datetime, timedelta
from functools import wraps
from src.config.settings import settings
from src.utils.error_handlers import AppException, ErrorCode
from src.utils.logging_config import log_audit_event
import hashlib
import secrets

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET = settings.secret_key
JWT_ALGORITHM = settings.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes

# Security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


class SecurityMiddleware:
    """
    Security middleware for the application
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))

                # Add security headers
                for header, value in SECURITY_HEADERS.items():
                    headers[header.encode()] = value.encode()

                # Add custom security headers
                headers[b"X-Content-Type-Options"] = b"nosniff"
                headers[b"X-Frame-Options"] = b"DENY"

                message["headers"] = [(k, v) for k, v in headers.items()]

            await send(message)

        await self.app(scope, receive, send_wrapper)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a JWT token and return the payload
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        # Check if token is expired
        exp = payload.get("exp")
        if exp and exp < time.time():
            logger.warning("Token expired")
            return None

        return payload
    except jwt.JWTError as e:
        logger.error(f"Token verification error: {e}")
        return None


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))):
    """
    Get current user from JWT token
    """
    if not credentials:
        return None

    token = credentials.credentials
    user_data = verify_token(token)

    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_data


def require_auth(required_roles: Optional[list] = None):
    """
    Decorator to require authentication and optionally specific roles
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract credentials from request
            request = kwargs.get('request') or next((arg for arg in args if isinstance(arg, Request)), None)

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Request object is required"
                )

            # Check for authorization header
            auth_header = request.headers.get("authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authorization header missing or invalid format"
                )

            token = auth_header[7:]  # Remove "Bearer " prefix
            user_data = verify_token(token)

            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )

            # Check roles if required
            if required_roles:
                user_roles = user_data.get("roles", [])
                has_required_role = any(role in user_roles for role in required_roles)

                if not has_required_role:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )

            # Add user data to kwargs for the decorated function
            kwargs['current_user'] = user_data

            # Log the access for audit purposes
            log_audit_event(
                event_type="auth_access",
                user_id=user_data.get("user_id", "unknown"),
                resource=request.url.path,
                action=request.method,
                success=True
            )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimiter:
    """
    Rate limiting implementation
    """

    def __init__(self):
        self.requests = {}
        self.blocked_ips = {}
        self.window_size = settings.rate_limit_window  # seconds
        self.max_requests = settings.rate_limit_requests

    def is_allowed(self, identifier: str) -> bool:
        """
        Check if the request is allowed based on rate limits
        """
        current_time = time.time()

        # Check if IP is blocked
        if identifier in self.blocked_ips:
            block_until = self.blocked_ips[identifier]
            if current_time < block_until:
                return False
            else:
                # Unblock if time has passed
                del self.blocked_ips[identifier]

        # Get or initialize request history
        if identifier not in self.requests:
            self.requests[identifier] = []

        # Clean old requests outside the window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.window_size
        ]

        # Check if limit exceeded
        if len(self.requests[identifier]) >= self.max_requests:
            # Block the IP for a period (e.g., 10 minutes)
            self.blocked_ips[identifier] = current_time + 600  # 10 minutes
            return False

        # Add current request
        self.requests[identifier].append(current_time)

        return True

    def get_remaining_requests(self, identifier: str) -> tuple:
        """
        Get remaining requests and reset time for an identifier
        """
        current_time = time.time()

        if identifier not in self.requests:
            return self.max_requests, current_time + self.window_size

        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.window_size
        ]

        remaining = max(0, self.max_requests - len(self.requests[identifier]))
        reset_time = current_time + (self.window_size - (current_time % self.window_size))

        return remaining, reset_time


# Global rate limiter instance
rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """
    Middleware for rate limiting
    """
    # Skip rate limiting for health check endpoints
    if request.url.path in ["/health", "/ready", "/live", "/metrics", "/docs", "/redoc", "/openapi.json"]:
        response = await call_next(request)
        return response

    # Get client IP
    client_ip = request.client.host

    # Check if request is allowed
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")

        # Log the violation
        log_audit_event(
            event_type="rate_limit_violation",
            user_id="unknown",
            resource=request.url.path,
            action=request.method,
            success=False,
            details={"client_ip": client_ip}
        )

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )

    # Add rate limit headers
    remaining, reset_time = rate_limiter.get_remaining_requests(client_ip)

    response = await call_next(request)

    # Add rate limit headers to response
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(int(reset_time))

    return response


def hash_password(password: str) -> str:
    """
    Hash a password for storage
    """
    salt = secrets.token_hex(16)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return salt + pwdhash.hex()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash
    """
    salt = hashed_password[:32]
    stored_hash = hashed_password[32:]
    pwdhash = hashlib.pbkdf2_hmac('sha256', plain_password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return pwdhash.hex() == stored_hash


def setup_security_middleware(app):
    """
    Setup security middleware for the application
    """
    # Add security headers middleware
    app.add_middleware(SecurityMiddleware)

    # Add rate limiting middleware
    app.middleware("http")(rate_limit_middleware)


# Authentication utility functions
def create_user_token(user_id: str, roles: list = None) -> str:
    """
    Create a token for a user with specific roles
    """
    if roles is None:
        roles = ["user"]

    data = {
        "user_id": user_id,
        "roles": roles,
        "iat": datetime.utcnow(),
        "sub": user_id
    }

    return create_access_token(data, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))


def require_role(role: str):
    """
    Decorator to require a specific role
    """
    return require_auth(required_roles=[role])


def require_any_role(roles: list):
    """
    Decorator to require any of the specified roles
    """
    return require_auth(required_roles=roles)


# Admin role decorator
require_admin = require_role("admin")
require_user = require_role("user")


logger.info("Security middleware initialized")