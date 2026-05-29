from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import torch
import torchaudio


_FFMPEG_BIN = shutil.which("ffmpeg")


def _decode_with_torchaudio(raw: bytes) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(io.BytesIO(raw))
    return wav, sr


def _decode_with_ffmpeg(raw: bytes) -> Tuple[torch.Tensor, int]:
    if not _FFMPEG_BIN:
        raise RuntimeError("未找到 ffmpeg 可执行文件。")

    cmd = [
        _FFMPEG_BIN,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=raw, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or f"ffmpeg exited with code {proc.returncode}")

    if not proc.stdout:
        raise RuntimeError("ffmpeg 未输出音频数据。")

    wav = torch.frombuffer(bytearray(proc.stdout), dtype=torch.int16).float() / 32768.0
    return wav.unsqueeze(0), 16000


def _needs_ffmpeg_fallback(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in {".mp3", ".aac", ".m4a", ".mp4", ".oga", ".opus"}


def load_audio_bytes(raw: bytes, filename: str) -> Tuple[torch.Tensor, int]:
    """
    Decode uploaded audio bytes to mono waveform tensor (1, T) and sample rate.

    Uses torchaudio for lossless/common containers; falls back to FFmpeg for
    compressed formats such as MP3/AAC/M4A when torchaudio backends cannot decode.
    """
    if not raw:
        raise ValueError("音频内容为空，请重新上传有效文件。")

    errors: list[str] = []
    if not _needs_ffmpeg_fallback(filename):
        try:
            return _decode_with_torchaudio(raw)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"torchaudio: {exc}")

    if _FFMPEG_BIN:
        try:
            return _decode_with_ffmpeg(raw)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ffmpeg: {exc}")
    else:
        errors.append("ffmpeg: 未安装")

    if not _needs_ffmpeg_fallback(filename):
        try:
            return _decode_with_ffmpeg(raw)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ffmpeg fallback: {exc}")

    detail = "；".join(errors) if errors else "未知解码错误"
    raise ValueError(
        f"音频解码失败（{Path(filename).name}）：{detail}。"
        "无损或常见开源容器（WAV、FLAC、OGG）通常在装有 libsndfile 的环境下即可解码；"
        "MP3、AAC、M4A 等需要服务器安装 FFmpeg。"
    )
