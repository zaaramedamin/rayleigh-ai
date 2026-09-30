import json
import logging

from app.core.logging import JsonFormatter


def test_json_formatter_emits_valid_json() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello %s", ("world",), None)

    entry = json.loads(JsonFormatter().format(record))

    assert entry["level"] == "INFO"
    assert entry["logger"] == "test"
    assert entry["message"] == "hello world"
    assert "time" in entry
