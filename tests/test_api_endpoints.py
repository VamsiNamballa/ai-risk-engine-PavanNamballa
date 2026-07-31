"""
End-to-end integration tests for FastAPI HTTP API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from service.main import app

client = TestClient(app)


def test_config_endpoint():
    """Verify /config endpoint returns active model name."""
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data


def test_home_endpoint():
    """Verify root GET / endpoint serves static HTML frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Risk Engine" in response.text or "<!DOCTYPE html>" in response.text


def test_upload_non_pdf_rejection():
    """Verify uploading non-PDF file returns 400 Bad Request error."""
    files = {"file": ("test.txt", b"Hello world", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_ask_endpoint_with_session_id():
    """Verify /ask endpoint accepts query and session_id header/payload."""
    payload = {"query": "What is the company risk level?", "session_id": "test_session_api"}
    response = client.post("/ask", json=payload, headers={"X-Session-ID": "test_session_api"})
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "executive_summary" in data
    assert data["session_id"] == "test_session_api"
