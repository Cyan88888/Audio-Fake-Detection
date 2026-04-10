"""
FastAPI service: upload audio -> spoof detection + waveform / log-mel for visualization.

Start (from repo root, after training/export):
  export SAFEAR_CKPT=Exps/TransformerSpoof19_hubert_e30/checkpoints/last.ckpt
  uvicorn web.api:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference.hubert_featurizer import HubertFeaturizer
from inference.load_model import load_detector_auto

_WEB_DIR = Path(__file__).resolve().parent

_detector: Optional[nn.Module] = None
_featurizer: Optional[HubertFeaturizer] = None
_device: Optional[torch.device] = None


def _get_device() -> torch.device:
    return torch.device(os.environ.get("SAFEAR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector, _featurizer, _device
    _device = _get_device()
    ckpt = os.environ.get("SAFEAR_CKPT")
    hubert = os.environ.get("SAFEAR_HUBERT", str(_ROOT / "model_zoos" / "hubert_base_ls960.pt"))
    if ckpt and Path(ckpt).is_file():
        _detector = load_detector_auto(ckpt, map_location=str(_device))
        _detector.eval()
        _detector.to(_device)
        _featurizer = HubertFeaturizer(ckpt_path=hubert, device=_device)
    yield


app = FastAPI(title="Transformer Spoof Detection", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _wav_mel_for_plot(wav_1d: torch.Tensor, sr: int, max_wave_points: int = 4000) -> Dict[str, Any]:
    """Downsampled waveform + log-mel (dB) for browser visualization."""
    wav = wav_1d.float()
    if wav.numel() > max_wave_points:
        step = wav.numel() // max_wave_points
        wav_ds = wav[::step][:max_wave_points]
    else:
        wav_ds = wav
    wave_list = wav_ds.cpu().numpy().tolist()

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=160, n_mels=64, mel_scale="htk"
    )(wav.unsqueeze(0))
    mel_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)(mel)
    z = mel_db.squeeze(0).numpy()
    # limit time bins for JSON size
    max_t = 200
    if z.shape[1] > max_t:
        z = z[:, :max_t]
    return {"waveform": wave_list, "mel_db": z.tolist(), "sample_rate": sr}


@app.get("/health")
def health():
    ok = _detector is not None and _featurizer is not None
    return {"status": "ok" if ok else "no_model", "ckpt": os.environ.get("SAFEAR_CKPT")}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), max_len: int = 64600):
    if _detector is None or _featurizer is None:
        raise HTTPException(
            503,
            detail="Model not loaded. Set SAFEAR_CKPT to a .ckpt or exported .pt and restart.",
        )
    raw = await file.read()
    buf = io.BytesIO(raw)
    try:
        wav, sr = torchaudio.load(buf)
    except Exception as e:
        raise HTTPException(400, detail=f"Could not decode audio: {e}") from e
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = AF.resample(wav, sr, 16000)
        sr = 16000
    wav_1d = wav.squeeze(0)
    plot_payload = _wav_mel_for_plot(wav_1d, sr)

    feat = _featurizer.wav_tensor_to_feat(wav_1d, max_len=max_len).to(_device)
    with torch.no_grad():
        logits, _ = _detector(feat)
        prob = torch.softmax(logits, dim=-1)[0]
        pred = int(torch.argmax(logits, dim=-1).item())

    return JSONResponse(
        {
            "filename": file.filename,
            "prob_bonafide": float(prob[0].item()),
            "prob_spoof": float(prob[1].item()),
            "pred_label": "bonafide" if pred == 0 else "spoof",
            "pred_class": pred,
            **plot_payload,
        }
    )


@app.get("/")
def index_page():
    p = _WEB_DIR / "static" / "index.html"
    if not p.is_file():
        return JSONResponse({"error": "static/index.html missing"}, status_code=404)
    return FileResponse(p)


static_dir = _WEB_DIR / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
