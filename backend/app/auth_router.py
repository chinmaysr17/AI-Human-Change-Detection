
import os
from datetime import timedelta

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    Cookie,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.auth_service import (
    register_user,
    authenticate_user,
    create_user_session,
    get_user_by_session_token,
    revoke_user_session,
)


router = APIRouter(prefix="/api/auth", tags=["Authentication"])

SESSION_COOKIE_NAME = "hcd_session"
SESSION_MAX_AGE = 86400

FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "http://127.0.0.1:5501",
)

COOKIE_SECURE = (
    os.getenv("COOKIE_SECURE", "false").lower() == "true"
)


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=150)
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=1, max_length=128)


def verify_frontend_origin(request: Request):
    """Validate the browser origin for state-changing requests."""

    origin = request.headers.get("origin")

    if origin != FRONTEND_ORIGIN:
        raise HTTPException(
            status_code=403,
            detail="Invalid request origin",
        )


def get_current_user(
    hcd_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    """Validate the session cookie and return the current user."""

    user = get_user_by_session_token(db, hcd_session or "")

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    return user


@router.post("/register", status_code=201)
def register(
    payload: RegisterRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    verify_frontend_origin(request)

    try:
        user = register_user(
            db,
            payload.full_name,
            payload.email,
            payload.password,
        )

        return {
            "message": "Registration successful",
            "user": {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
            },
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )


@router.post("/login")
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    verify_frontend_origin(request)

    user = authenticate_user(
        db,
        payload.email,
        payload.password,
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    token = create_user_session(db, user)

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )

    return {
        "message": "Login successful",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
        },
    }


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    hcd_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    verify_frontend_origin(request)

    if hcd_session:
        revoke_user_session(db, hcd_session)

    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
    )

    return {"message": "Logout successful"}


@router.get("/me")
def get_me(
    current_user=Depends(get_current_user),
):
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
    }
