from .audit import router as audit_router
from .auth import router as auth_router
from .health import router as health_router
from .history import router as history_router
from .predict import router as predict_router
from .tasks import router as task_router

__all__ = [
    "audit_router",
    "auth_router",
    "health_router",
    "history_router",
    "predict_router",
    "task_router",
]
