import threading

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.access import password as passwords
from app.access.sessions import sessions, throttle
from app.api.access import bearer_token
from app.api.deps import SettingsDep

router = APIRouter(prefix="/auth", tags=["auth"])

_SETUP_LOCK = threading.Lock()


class AuthStatus(BaseModel):
    required: bool
    configured: bool = Field(description="A password has been chosen.")
    authenticated: bool


class Credentials(BaseModel):
    password: str = Field(min_length=1, max_length=passwords.MAX_PASSWORD_LENGTH)


class TokenResponse(BaseModel):
    token: str


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=passwords.MAX_PASSWORD_LENGTH)
    new_password: str = Field(min_length=1, max_length=passwords.MAX_PASSWORD_LENGTH)


def _refuse_if_throttled() -> None:
    wait = throttle.seconds_to_wait()
    if wait:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Too many wrong passwords. Try again in {wait} seconds.",
            headers={"Retry-After": str(wait)},
        )


@router.get("/status")
def auth_status(request: Request, settings: SettingsDep) -> AuthStatus:
    """Whether a password exists and whether this request is signed in. Needs no sign-in."""
    token = bearer_token(request)
    return AuthStatus(
        required=settings.access_required,
        configured=passwords.is_configured(settings.data_dir),
        authenticated=(not settings.access_required)
        or (token is not None and sessions.validate(token)),
    )


@router.post("/setup", status_code=status.HTTP_201_CREATED)
def setup(credentials: Credentials, settings: SettingsDep) -> TokenResponse:
    """Choose the access password the first time. Refused once one exists."""
    with _SETUP_LOCK:
        if passwords.is_configured(settings.data_dir):
            raise HTTPException(status.HTTP_409_CONFLICT, "A password has already been chosen.")
        try:
            passwords.set_password(settings.data_dir, credentials.password)
        except passwords.PasswordError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return TokenResponse(token=sessions.create())


@router.post("/login")
def login(credentials: Credentials, settings: SettingsDep) -> TokenResponse:
    _refuse_if_throttled()
    if not passwords.verify_password(settings.data_dir, credentials.password):
        throttle.record_failure()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong password.")
    throttle.record_success()
    return TokenResponse(token=sessions.create())


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request) -> None:
    token = bearer_token(request)
    if token is not None:
        sessions.revoke(token)


@router.post("/change-password")
def change_password(
    change: PasswordChange, request: Request, settings: SettingsDep
) -> TokenResponse:
    """Replace the password. Needs the current one, and signs out every other session."""
    token = bearer_token(request)
    if settings.access_required and (token is None or not sessions.validate(token)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in required.")
    _refuse_if_throttled()
    if not passwords.verify_password(settings.data_dir, change.current_password):
        throttle.record_failure()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "The current password is wrong.")
    try:
        passwords.set_password(settings.data_dir, change.new_password)
    except passwords.PasswordError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    throttle.record_success()
    sessions.clear()
    return TokenResponse(token=sessions.create())
