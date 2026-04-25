from __future__ import annotations

from datetime import datetime, timezone

from ..db.models import AuditRecord
from ..db.repository import repository


def log_action(action: str, actor: str, detail: str) -> None:
    repository.insert_audit(
        AuditRecord(
            id=None,
            action=action,
            actor=actor,
            detail=detail,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    )
