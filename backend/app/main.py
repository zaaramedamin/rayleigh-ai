import logging

from fastapi import FastAPI

from app import __version__
from app.api.v1 import health
from app.core.config import get_settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="Reyleight", version=__version__)
    app.include_router(health.router, prefix="/api/v1")

    logger.info("app started env=%s", settings.app_env)
    return app


app = create_app()
