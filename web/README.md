# SafeEar Web System (FastAPI + React)

This module provides a lightweight audio spoof detection backend plus React frontend.

## 1) Backend architecture

- Entry: `web/api.py`
- Routes:
  - `web/routes/health.py`
  - `web/routes/predict.py`
  - `web/routes/tasks.py`
- Services:
  - `web/services/inference_service.py`
  - `web/services/task_service.py`

## 2) Frontend architecture

- React app root: `web/frontend/`
- Main page: `web/frontend/src/App.jsx`
- Styling: `web/frontend/src/styles.css`

## 3) Features

- Single and batch inference (`/api/predict`, `/api/predict_batch`)
- Task tracking (`/api/tasks/{job_id}`)
- Fixed backend threshold decision (`SAFEAR_WEB_FIXED_THRESHOLD_SPOOF`)
- Waveform and Mel-spectrogram payload for frontend display

## 4) Run backend with HuBERT

```bash
cd /root/autodl-tmp/SafeEar
source /root/miniconda3/etc/profile.d/conda.sh
conda activate safeear

export SAFEAR_CKPT=Exps/Search_PoolMax_S3_ls002_hubert/checkpoints/epoch=4-val_eer=0.0009.ckpt
export SAFEAR_FEAT=hubert
export SAFEAR_HUBERT=model_zoos/hubert_base_ls960.pt
export SAFEAR_WEB_FIXED_THRESHOLD_SPOOF=0.5

uvicorn web.api:app --host 0.0.0.0 --port 8080
```

## 5) Run frontend (React dev)

```bash
cd /root/autodl-tmp/SafeEar/web/frontend
npm install
npm run dev
```

For production static assets:

```bash
cd /root/autodl-tmp/SafeEar/web/frontend
npm run build
```

Then restart backend. `web/api.py` will serve `web/frontend/dist/index.html` automatically.

## 6) API quick reference

- `GET /health`
- `POST /api/predict` form-data: `file`, optional `threshold`, `max_len`
- `POST /api/predict_batch` form-data: `files[]`, optional `threshold`, `max_len`
- `GET /api/tasks/{job_id}`

## 7) 支持的音频格式与解码环境

- 服务端按**文件扩展名**白名单校验，允许的 suffix 定义见 `web/audio_formats.py`（含 `.wav`、`.flac`、`.ogg`、`.mp3`、`.aac`、`.m4a`、`.mp4` 等）。
- 实际能否解码由 `torchaudio.load` 与系统解码后端决定：**无损或常见开源容器**（WAV、FLAC、OGG）在装有相应库的环境下通常可直接解码。
- **MP3、AAC、M4A** 等压缩格式在多数 Linux 环境下需要安装 **FFmpeg**，并确保 torchaudio 构建或运行时能使用该后端；否则会返回 **400** 及明确错误说明。
- 若解码失败，接口返回 `400`，`detail` 中会提示环境与 FFmpeg 相关信息。
