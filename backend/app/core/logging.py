import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON.

    Only log metadata and event names here. Never log document or query text.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
