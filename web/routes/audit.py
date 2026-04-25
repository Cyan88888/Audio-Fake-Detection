from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..db.repository import repository
from ..services.auth_service import require_token

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("")
def list_audit(limit: int = Query(default=100, ge=1, le=2000), actor: str = Depends(require_token)):
    _ = actor
    rows = repository.list_audit(limit=limit)
    return {"items": [dict(r) for r in rows]}
