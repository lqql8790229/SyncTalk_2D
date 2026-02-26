"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from synctalk.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert isinstance(data["gpu_available"], bool)

    def test_docs(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200


class TestInferenceEndpoints:
    def test_get_nonexistent_task(self, client):
        resp = client.get("/api/v1/inference/nonexistent-id")
        assert resp.status_code == 404

    def test_download_nonexistent(self, client):
        resp = client.get("/api/v1/inference/nonexistent-id/video")
        assert resp.status_code == 404


class TestProjectEndpoints:
    def test_list_projects(self, client):
        resp = client.get("/api/v1/projects")
        assert resp.status_code == 200
        assert "projects" in resp.json()

    def test_model_info(self, client):
        resp = client.get("/api/v1/models/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "160px" in data
        assert "328px" in data
        assert data["328px"]["total_params"] == 12_186_935
        assert data["160px"]["total_params"] == 12_163_591
