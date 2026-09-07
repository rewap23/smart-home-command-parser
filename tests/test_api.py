from fastapi.testclient import TestClient

from smart_home_parser.api import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_ok() -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata_returns_expected_fields() -> None:
    response = client.get("/metadata")

    assert response.status_code == 200

    body = response.json()
    assert body["app_name"] == "Smart Home Command Parser API"
    assert body["app_version"] == "0.1.0"
    assert body["model_loaded"] is False
