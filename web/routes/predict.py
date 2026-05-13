from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from .. import config
from ..audio_formats import validate_audio_upload_filename
from ..schemas.predict import PredictResponse
from ..services.inference_service import inference_service
from ..services.task_service import task_service

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict_one(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    max_len: int = Form(64600),
):
    _ = threshold  # Kept for backward-compatible form input; runtime threshold is fixed by config.
    if not inference_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded.")
    fixed_threshold = config.get_web_fixed_threshold_spoof()
    raw = await file.read()
    try:
        item = inference_service.predict_bytes(
            raw=raw,
            filename=file.filename or "upload.wav",
            max_len=max_len,
            threshold=fixed_threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    job_id = inference_service.new_job_id()
    return PredictResponse(
        job_id=job_id,
        threshold=fixed_threshold,
        model_version=inference_service.model_version,
        items=[item],
    )


@router.post("/predict_batch")
async def predict_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    threshold: float = Form(0.5),
    max_len: int = Form(64600),
):
    _ = threshold  # Kept for backward-compatible form input; runtime threshold is fixed by config.
    if not inference_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    file_payloads = []
    for f in files:
        name = f.filename or ""
        try:
            validate_audio_upload_filename(name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        file_payloads.append(
            {
                "filename": name or "upload.wav",
                "raw": await f.read(),
            }
        )
    job_id = task_service.enqueue_batch(
        bg=background_tasks,
        files=file_payloads,
        threshold=config.get_web_fixed_threshold_spoof(),
        max_len=max_len,
    )
    return {"job_id": job_id, "status": "pending", "total_files": len(files)}
