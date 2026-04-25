from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from ..schemas.predict import PredictResponse
from ..services.auth_service import require_token
from ..services.audit_service import log_action
from ..services.history_service import save_prediction_items
from ..services.inference_service import inference_service
from ..services.task_service import task_service

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict_one(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    max_len: int = Form(64600),
    actor: str = Depends(require_token),
):
    if not inference_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded.")
    raw = await file.read()
    item = inference_service.predict_bytes(
        raw=raw,
        filename=file.filename or "upload.wav",
        max_len=max_len,
        threshold=threshold,
    )
    job_id = inference_service.new_job_id()
    save_prediction_items(job_id=job_id, items=[item])
    log_action("predict_one", actor, f"job={job_id} file={item['filename']}")
    return PredictResponse(
        job_id=job_id,
        threshold=threshold,
        model_version=inference_service.model_version,
        items=[item],
    )


@router.post("/predict_batch")
async def predict_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    threshold: float = Form(0.5),
    max_len: int = Form(64600),
    actor: str = Depends(require_token),
):
    if not inference_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    file_payloads = []
    for f in files:
        file_payloads.append(
            {
                "filename": f.filename or "upload.wav",
                "raw": await f.read(),
            }
        )
    job_id = task_service.enqueue_batch(
        bg=background_tasks,
        files=file_payloads,
        threshold=threshold,
        max_len=max_len,
        actor=actor,
    )
    return {"job_id": job_id, "status": "pending", "total_files": len(files)}
