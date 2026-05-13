"""
Declared upload extensions for the web API.

Decoding is performed by ``torchaudio.load``; actual codec support depends on the
runtime (e.g. FFmpeg for many compressed formats). Extension checks only enforce
a clear contract with clients.
"""

from __future__ import annotations

from pathlib import Path

# Lowercase suffixes including common aliases.
ALLOWED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".wav",
        ".wave",
        ".flac",
        ".ogg",
        ".oga",
        ".opus",
        ".mp3",
        ".aac",
        ".m4a",
        ".mp4",  # some M4A tracks use .mp4 container
    }
)

_ALLOWED_DISPLAY = "WAV、FLAC、OGG（含 Opus）、MP3、AAC、M4A"


def validate_audio_upload_filename(filename: str) -> None:
    """
    Raise ValueError with a user-facing message if the suffix is missing or not allowed.
    """
    name = (filename or "").strip()
    if not name:
        raise ValueError("文件名为空，请上传带扩展名的音频文件。")

    suf = Path(name).suffix.lower()
    if not suf:
        raise ValueError(
            f"文件名缺少扩展名。服务端按扩展名校验格式，请使用 {_ALLOWED_DISPLAY} 等常见后缀。"
        )

    if suf not in ALLOWED_AUDIO_EXTENSIONS:
        allowed_sorted = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise ValueError(
            f"不支持的文件扩展名「{suf}」。允许的扩展名：{allowed_sorted}。"
            "解码仍依赖运行环境（压缩格式通常需要 FFmpeg）。"
        )
