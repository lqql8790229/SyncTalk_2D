"""Tests for Python SDK client."""

import pytest
from fastapi.testclient import TestClient
from synctalk.api.app import app
from synctalk.sdk import SyncTalkClient


@pytest.fixture
def client():
    return TestClient(app)


class TestSDKWithTestClient:
    """Test SDK methods using FastAPI TestClient as backend."""

    def test_health(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"

    def test_list_projects(self, client):
        resp = client.get("/api/v1/projects")
        assert resp.status_code == 200
        assert "projects" in resp.json()

    def test_model_info(self, client):
        resp = client.get("/api/v1/models/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["328px"]["total_params"] == 12_186_935

    def test_inference_nonexistent(self, client):
        resp = client.get("/api/v1/tasks/nonexistent")
        assert resp.status_code == 404


class TestSDKClientInit:
    def test_create_client(self):
        sdk = SyncTalkClient("http://localhost:8000")
        assert sdk.base_url == "http://localhost:8000"

    def test_create_with_api_key(self):
        sdk = SyncTalkClient("http://localhost:8000", api_key="test-key")
        assert sdk.session.headers["X-API-Key"] == "test-key"

    def test_url_trailing_slash(self):
        sdk = SyncTalkClient("http://localhost:8000/")
        assert sdk.base_url == "http://localhost:8000"
