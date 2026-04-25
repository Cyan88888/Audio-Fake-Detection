from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HistoryRecord:
    id: Optional[int]
    job_id: str
    filename: str
    pred_label: str
    prob_spoof: float
    prob_bonafide: float
    threshold: float
    decision_by_threshold: str
    model_version: str
    inference_time_ms: float
    created_at: str
    payload_json: str


@dataclass
class AuditRecord:
    id: Optional[int]
    action: str
    actor: str
    detail: str
    created_at: str
