from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List

from fastapi import BackgroundTasks

from .audit_service import log_action
from .history_service import save_prediction_items
from .inference_service import inference_service


class TaskService:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def create_job(self, total_files: int) -> str:
        job_id = inference_service.new_job_id()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "created_at": now,
                "completed_at": None,
                "total_files": total_files,
                "done_files": 0,
                "items": [],
                "error": None,
            }
        return job_id

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            data = self._jobs.get(job_id)
            if not data:
                raise KeyError(job_id)
            return dict(data)

    def _update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            self._jobs[job_id].update(kwargs)

    async def run_batch(
        self,
        job_id: str,
        files: List[Dict[str, Any]],
        threshold: float,
        max_len: int,
        actor: str,
    ) -> None:
        self._update(job_id, status="running")
        items: List[Dict[str, Any]] = []
        try:
            for idx, file_obj in enumerate(files, start=1):
                raw = file_obj["raw"]
                filename = file_obj.get("filename") or f"file_{idx}"
                item = inference_service.predict_bytes(
                    raw=raw,
                    filename=filename,
                    max_len=max_len,
                    threshold=threshold,
                )
                items.append(item)
                self._update(job_id, done_files=idx)
            save_prediction_items(job_id=job_id, items=items)
            now = datetime.now(timezone.utc).isoformat()
            self._update(job_id, status="completed", completed_at=now, items=items)
            log_action("predict_batch", actor, f"job={job_id} files={len(items)}")
        except Exception as e:  # noqa: BLE001
            now = datetime.now(timezone.utc).isoformat()
            self._update(job_id, status="failed", completed_at=now, error=str(e))
            log_action("predict_batch_failed", actor, f"job={job_id} error={e}")

    def enqueue_batch(
        self,
        bg: BackgroundTasks,
        files: List[Dict[str, Any]],
        threshold: float,
        max_len: int,
        actor: str,
    ) -> str:
        job_id = self.create_job(total_files=len(files))
        bg.add_task(self.run_batch, job_id, files, threshold, max_len, actor)
        return job_id


task_service = TaskService()
