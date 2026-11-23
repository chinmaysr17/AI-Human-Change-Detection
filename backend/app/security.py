
import hashlib
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)


# Argon2 password hashing configuration
password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Generate a secure Argon2 password hash."""
    if not password:
        raise ValueError("Password cannot be empty")

    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its stored Argon2 hash."""
    try:
        return password_hasher.verify(password_hash, password)
    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHashError,
    ):
        return False


def create_session_token() -> str:
    """Generate a cryptographically secure session token."""
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    """Hash a session token before storing it in PostgreSQL."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
