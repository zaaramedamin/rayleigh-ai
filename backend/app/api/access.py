"""The sign-in check that protects every route except /health and /auth."""

from fastapi import HTTPException, Request, status

from app.access.sessions import sessions
from app.api.deps import SettingsDep


def bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def require_access(request: Request, settings: SettingsDep) -> None:
    """Refuse the request unless it carries a valid sign-in token."""
    if not settings.access_required:
        return
    token = bearer_token(request)
    if token is None or not sessions.validate(token):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Sign in required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
