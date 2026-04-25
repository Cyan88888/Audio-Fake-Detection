"""
FastAPI entrypoint for S2 web system.

Start:
  export SAFEAR_CKPT=Exps/Ablation_S2_pe_learnable/checkpoints/epoch=7-val_eer=0.0247.ckpt
  uvicorn web.api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .routes import audit_router, auth_router, health_router, history_router, predict_router, task_router
from .services.inference_service import inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = app
    inference_service.load()
    yield


app = FastAPI(title="S2 Audio Spoof Detection System", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(task_router)
app.include_router(history_router)
app.include_router(audit_router)


@app.get("/")
def index_page():
    react_dist = config.WEB_DIR / "frontend" / "dist" / "index.html"
    static_html = config.WEB_DIR / "static" / "index.html"
    if react_dist.is_file():
        return FileResponse(react_dist)
    if static_html.is_file():
        return FileResponse(static_html)
    return JSONResponse({"error": "No frontend found"}, status_code=404)


react_static = config.WEB_DIR / "frontend" / "dist" / "assets"
legacy_static = config.WEB_DIR / "static"
if react_static.is_dir():
    app.mount("/assets", StaticFiles(directory=str(react_static)), name="assets")
if legacy_static.is_dir():
    app.mount("/static", StaticFiles(directory=str(legacy_static)), name="static")
