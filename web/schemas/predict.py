from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictItem(BaseModel):
    filename: str
    prob_bonafide: float
    prob_spoof: float
    pred_label: str
    pred_class: int
    sample_rate: int
    waveform: List[float]
    mel_db: List[List[float]]
    threshold: float
    decision_by_threshold: str
    confidence_gap: float
    inference_time_ms: float
    created_at: str
    model_version: str


class PredictResponse(BaseModel):
    job_id: str
    threshold: float
    model_version: str
    items: List[PredictItem]
    error: Optional[str] = None


class BatchJobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    total_files: int = 0
    done_files: int = 0
    error: Optional[str] = None
    result_path: Optional[str] = None


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
