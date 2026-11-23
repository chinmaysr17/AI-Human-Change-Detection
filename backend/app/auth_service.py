
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import User, UserSession
from backend.app.security import (
    hash_password,
    verify_password,
    create_session_token,
    hash_session_token,
)


# Login session duration: 1 day (24 hours)
SESSION_DURATION = timedelta(days=1)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def register_user(
    db: Session,
    full_name: str,
    email: str,
    password: str,
) -> User:
    """Register a new user."""

    full_name = full_name.strip()
    email = normalize_email(email)

    if not full_name:
        raise ValueError("Full name is required")

    if len(full_name) > 150:
        raise ValueError("Full name is too long")

    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters")

    existing_user = db.scalar(
        select(User).where(User.email == email)
    )

    if existing_user:
        raise ValueError("Email is already registered")

    user = User(
        full_name=full_name,
        email=email,
        password_hash=hash_password(password),
    )

    try:
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    except IntegrityError:
        db.rollback()
        raise ValueError("Email is already registered")


def authenticate_user(
    db: Session,
    email: str,
    password: str,
) -> User | None:
    """Verify user credentials."""

    email = normalize_email(email)

    user = db.scalar(
        select(User).where(User.email == email)
    )

    if user is None:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user


def create_user_session(
    db: Session,
    user: User,
) -> str:
    """Create a 24-hour session and return its raw token."""

    token = create_session_token()
    token_hash = hash_session_token(token)

    now = datetime.now(timezone.utc)

    session = UserSession(
        user_id=user.id,
        token_hash=token_hash,
        created_at=now,
        expires_at=now + SESSION_DURATION,
    )

    db.add(session)
    db.commit()

    return token


def get_user_by_session_token(
    db: Session,
    token: str,
) -> User | None:
    """Validate a session token and return its user."""

    if not token:
        return None

    token_hash = hash_session_token(token)
    now = datetime.now(timezone.utc)

    session = db.scalar(
        select(UserSession).where(
            UserSession.token_hash == token_hash,
            UserSession.revoked_at.is_(None),
            UserSession.expires_at > now,
        )
    )

    if session is None:
        return None

    return db.get(User, session.user_id)


def revoke_user_session(
    db: Session,
    token: str,
) -> bool:
    """Revoke an active session during logout."""

    if not token:
        return False

    token_hash = hash_session_token(token)

    session = db.scalar(
        select(UserSession).where(
            UserSession.token_hash == token_hash,
            UserSession.revoked_at.is_(None),
        )
    )

    if session is None:
        return False

    session.revoked_at = datetime.now(timezone.utc)
    db.commit()

    return True
