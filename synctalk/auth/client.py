"""Client-side authentication manager.

Handles login, registration, token storage, and auto-refresh.
Stores credentials locally in ~/.synctalk/auth.json.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

AUTH_CACHE_DIR = Path.home() / ".synctalk"
AUTH_CACHE_FILE = AUTH_CACHE_DIR / "auth.json"


class AuthClient:
    """Client authentication manager.

    Args:
        server_url: Cloud API server URL.
    """

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip("/")
        self._token = None
        self._user_id = None
        self._email = None
        self._plan = None
        self._load_cached_auth()

    @property
    def is_authenticated(self) -> bool:
        return self._token is not None

    @property
    def token(self) -> Optional[str]:
        return self._token

    @property
    def user_id(self) -> Optional[str]:
        return self._user_id

    @property
    def plan(self) -> Optional[str]:
        return self._plan

    @property
    def email(self) -> Optional[str]:
        return self._email

    def register(self, email: str, password: str, display_name: str = "") -> dict:
        """Register a new account."""
        resp = requests.post(f"{self.server_url}/api/v1/auth/register", json={
            "email": email, "password": password, "display_name": display_name,
        }, timeout=10)
        if resp.status_code != 200:
            detail = resp.json().get("detail", "Registration failed")
            raise ValueError(detail)
        data = resp.json()
        self._save_auth(data)
        return data

    def login(self, email: str, password: str) -> dict:
        """Login with email and password."""
        resp = requests.post(f"{self.server_url}/api/v1/auth/login", json={
            "email": email, "password": password,
        }, timeout=10)
        if resp.status_code != 200:
            detail = resp.json().get("detail", "Login failed")
            raise ValueError(detail)
        data = resp.json()
        self._save_auth(data)
        return data

    def get_me(self) -> dict:
        """Get current user info."""
        resp = requests.get(
            f"{self.server_url}/api/v1/auth/me",
            headers=self._auth_headers(), timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def list_characters(self) -> list:
        """List user's characters from cloud."""
        resp = requests.get(
            f"{self.server_url}/api/v1/characters",
            headers=self._auth_headers(), timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def create_character(self, name: str, resolution: int = 328,
                          asr_mode: str = "ave") -> dict:
        """Create a new character entry on cloud."""
        resp = requests.post(
            f"{self.server_url}/api/v1/characters",
            json={"name": name, "resolution": resolution, "asr_mode": asr_mode},
            headers=self._auth_headers(), timeout=10,
        )
        if resp.status_code != 200:
            detail = resp.json().get("detail", "Create failed")
            raise ValueError(detail)
        return resp.json()

    def logout(self):
        """Clear local authentication."""
        self._token = None
        self._user_id = None
        self._email = None
        self._plan = None
        if AUTH_CACHE_FILE.exists():
            AUTH_CACHE_FILE.unlink()
        logger.info("Logged out")

    def _auth_headers(self) -> dict:
        if not self._token:
            raise ValueError("Not authenticated. Please login first.")
        return {"Authorization": f"Bearer {self._token}"}

    def _save_auth(self, data: dict):
        self._token = data["access_token"]
        self._user_id = data["user_id"]
        self._email = data["email"]
        self._plan = data["plan"]

        AUTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUTH_CACHE_FILE, "w") as f:
            json.dump({
                "access_token": self._token,
                "user_id": self._user_id,
                "email": self._email,
                "plan": self._plan,
                "saved_at": datetime.utcnow().isoformat(),
            }, f)
        logger.info(f"Authenticated as {self._email} ({self._plan})")

    def _load_cached_auth(self):
        if AUTH_CACHE_FILE.exists():
            try:
                with open(AUTH_CACHE_FILE) as f:
                    data = json.load(f)
                self._token = data.get("access_token")
                self._user_id = data.get("user_id")
                self._email = data.get("email")
                self._plan = data.get("plan")
                logger.info(f"Loaded cached auth: {self._email}")
            except Exception:
                logger.warning("Failed to load cached auth")
