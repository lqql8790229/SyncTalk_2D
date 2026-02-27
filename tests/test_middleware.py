"""Tests for API middleware."""

import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from synctalk.api.middleware import APIKeyMiddleware, RateLimitMiddleware


def _make_app(middleware_cls, **kwargs):
    test_app = FastAPI()

    @test_app.get("/health")
    async def health():
        return {"status": "ok"}

    @test_app.get("/api/test")
    async def test_endpoint():
        return {"data": "secret"}

    test_app.add_middleware(middleware_cls, **kwargs)
    return test_app


class TestAPIKeyMiddleware:
    def test_no_keys_configured_allows_all(self):
        app = _make_app(APIKeyMiddleware, api_keys=[])
        client = TestClient(app)
        resp = client.get("/api/test")
        assert resp.status_code == 200

    def test_with_valid_key(self):
        app = _make_app(APIKeyMiddleware, api_keys=["valid-key"])
        client = TestClient(app)
        resp = client.get("/api/test", headers={"X-API-Key": "valid-key"})
        assert resp.status_code == 200

    def test_with_invalid_key(self):
        app = _make_app(APIKeyMiddleware, api_keys=["valid-key"])
        client = TestClient(app)
        resp = client.get("/api/test", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_missing_key(self):
        app = _make_app(APIKeyMiddleware, api_keys=["valid-key"])
        client = TestClient(app)
        resp = client.get("/api/test")
        assert resp.status_code == 401

    def test_health_bypasses_auth(self):
        app = _make_app(APIKeyMiddleware, api_keys=["valid-key"])
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200


class TestRateLimitMiddleware:
    def test_within_limit(self):
        app = _make_app(RateLimitMiddleware, max_requests=5, window_seconds=60)
        client = TestClient(app)
        for _ in range(5):
            resp = client.get("/api/test")
            assert resp.status_code == 200

    def test_exceeds_limit(self):
        app = _make_app(RateLimitMiddleware, max_requests=3, window_seconds=60)
        client = TestClient(app)
        for _ in range(3):
            client.get("/api/test")
        resp = client.get("/api/test")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
