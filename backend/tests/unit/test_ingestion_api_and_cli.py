import pytest
from fastapi.testclient import TestClient

from app.knowledge.ingestion.__main__ import main
from app.knowledge.ingestion.file_types import supported_extensions
from app.main import app

client = TestClient(app)


def test_file_types_endpoint_lists_every_registered_extension() -> None:
    response = client.get("/api/v1/ingestion/file-types")

    assert response.status_code == 200
    listed = {ext for item in response.json() for ext in item["extensions"]}
    assert listed == set(supported_extensions())
    assert {"name", "extensions", "description"} <= response.json()[0].keys()


def test_cli_list_types_prints_supported_types(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--list-types"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Supported file types" in output
    assert ".md" in output
    assert ".json" in output
