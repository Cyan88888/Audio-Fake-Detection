from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..services.auth_service import require_token
from ..services.task_service import task_service

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("/{job_id}")
def get_task(job_id: str, actor: str = Depends(require_token)):
    _ = actor
    try:
        return task_service.get_job(job_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from e
