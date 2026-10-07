import logging

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.access import require_access
from app.api.guard import LocalOnlyGuard
from app.api.v1 import (
    ask,
    assistant,
    auth,
    chat,
    conversations,
    feedback,
    health,
    ingestion,
    library,
    profile,
    search,
    system,
    tasks,
    voice,
)
from app.core.config import get_settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="Reyleight", version=__version__)
    # /health and /auth are open; everything else needs the access password.
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(auth.router, prefix="/api/v1")
    protected = [Depends(require_access)]
    for router in (
        ingestion.router,
        search.router,
        ask.router,
        chat.router,
        conversations.router,
        feedback.router,
        system.router,
        tasks.router,
        library.router,
        profile.router,
        assistant.router,
        voice.router,
    ):
        app.include_router(router, prefix="/api/v1", dependencies=protected)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
            allow_credentials=False,
        )
    # Added last, so it runs first: a request from the wrong place is refused before anything else.
    app.add_middleware(
        LocalOnlyGuard,
        extra_hosts=settings.allowed_hosts,
        cors_origins=settings.cors_origins,
        max_body_bytes=settings.max_request_mb * 1024 * 1024,
        requests_per_minute=settings.rate_limit_per_minute,
    )

    @app.exception_handler(Exception)
    async def unexpected(_request: Request, exc: Exception) -> JSONResponse:
        # What went wrong can contain pieces of a note or a question, so only its kind is logged.
        logger.error("request failed error=%s", type(exc).__name__)
        return JSONResponse(
            {"detail": "Something went wrong on this computer. See the server log."},
            status_code=500,
        )

    logger.info("app started env=%s", settings.app_env)
    return app


app = create_app()
