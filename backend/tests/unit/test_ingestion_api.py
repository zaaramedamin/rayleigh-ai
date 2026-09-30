from fastapi.testclient import TestClient

from app.knowledge.ingestion.file_types import supported_extensions
from app.main import app

client = TestClient(app)


def test_file_types_endpoint_lists_every_registered_extension() -> None:
    response = client.get("/api/v1/ingestion/file-types")

    assert response.status_code == 200
    listed = {ext for item in response.json() for ext in item["extensions"]}
    assert listed == set(supported_extensions())
    assert {"name", "extensions", "description"} <= response.json()[0].keys()
