from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from ..services.auth_service import require_token
from ..services.history_service import export_history_csv, export_history_json, list_history

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("")
def history_list(limit: int = Query(default=100, ge=1, le=2000), actor: str = Depends(require_token)):
    _ = actor
    return {"items": list_history(limit=limit)}


@router.get("/export.csv")
def history_export_csv(limit: int = Query(default=100, ge=1, le=5000), actor: str = Depends(require_token)):
    _ = actor
    path = export_history_csv(limit=limit)
    return FileResponse(path, media_type="text/csv", filename=path.name)


@router.get("/export.json")
def history_export_json(limit: int = Query(default=100, ge=1, le=5000), actor: str = Depends(require_token)):
    _ = actor
    path = export_history_json(limit=limit)
    return FileResponse(path, media_type="application/json", filename=path.name)
