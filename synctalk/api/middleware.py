"""API middleware for authentication, CORS, and rate limiting."""

import os
import time
import logging
from collections import defaultdict
from typing import Optional

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware.

    Validates X-API-Key header against configured API keys.
    Skips auth for /health and /docs endpoints.
    """

    SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc",
                  "/docs/oauth2-redirect"}

    def __init__(self, app, api_keys: Optional[list] = None):
        super().__init__(app)
        self.api_keys = set(api_keys or [])
        env_keys = os.environ.get("SYNCTALK_API_KEYS", "")
        if env_keys:
            self.api_keys.update(k.strip() for k in env_keys.split(",") if k.strip())
        self.enabled = len(self.api_keys) > 0

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        api_key = request.headers.get(API_KEY_HEADER)
        if not api_key or api_key not in self.api_keys:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter.

    Limits requests per IP within a sliding time window.
    """

    def __init__(self, app, max_requests: int = RATE_LIMIT_REQUESTS,
                 window_seconds: int = RATE_LIMIT_WINDOW):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        self._requests[client_ip] = [
            t for t in self._requests[client_ip]
            if now - t < self.window
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(self.window)},
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests with timing."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed:.1f}ms)"
        )
        return response
