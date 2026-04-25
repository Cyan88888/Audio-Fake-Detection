from __future__ import annotations

from fastapi import APIRouter

from ..schemas.predict import LoginRequest, LoginResponse
from ..services.audit_service import log_action
from ..services.auth_service import login

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login_api(req: LoginRequest):
    token = login(req.username, req.password)
    log_action("login", req.username, "login success")
    return LoginResponse(access_token=token)
