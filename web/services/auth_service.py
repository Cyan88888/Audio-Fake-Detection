from __future__ import annotations

import os

from fastapi import Header, HTTPException

from .. import config


def _auth_enabled() -> bool:
    return str(os.environ.get("SAFEAR_WEB_REQUIRE_AUTH", "0")).lower() in {"1", "true", "yes"}


def login(username: str, password: str) -> str:
    if username == config.get_auth_user() and password == config.get_auth_password():
        return config.get_auth_token()
    raise HTTPException(status_code=401, detail="Invalid credentials")


def require_token(authorization: str = Header(default="")) -> str:
    if not _auth_enabled():
        return config.get_auth_user()
    expected = f"Bearer {config.get_auth_token()}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return config.get_auth_user()
