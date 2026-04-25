# S2 Web System (FastAPI + React)

This module upgrades the original single-file demo to a layered backend plus React frontend.

## 1) Backend architecture

- Entry: `web/api.py`
- Routes:
  - `web/routes/health.py`
  - `web/routes/auth.py`
  - `web/routes/predict.py`
  - `web/routes/tasks.py`
  - `web/routes/history.py`
  - `web/routes/audit.py`
- Services:
  - `web/services/inference_service.py`
  - `web/services/task_service.py`
  - `web/services/history_service.py`
  - `web/services/auth_service.py`
  - `web/services/audit_service.py`
- Persistence:
  - `web/db/repository.py` (SQLite, auto-created in `web/storage/web_app.db`)

## 2) Frontend architecture

- React app root: `web/frontend/`
- Main page: `web/frontend/src/App.jsx`
- Styling: `web/frontend/src/styles.css`

## 3) Features in this upgrade

- Single + batch inference (`/api/predict`, `/api/predict_batch`)
- Task tracking (`/api/tasks/{job_id}`)
- Threshold interaction (frontend slider + backend decision field)
- History and export (`/api/history`, `/api/history/export.csv`, `/api/history/export.json`)
- Basic auth token flow (`/api/auth/login`, Bearer token)
- Audit log (`/api/audit`)

## 4) Run backend

```bash
cd /root/autodl-tmp/SafeEar
source /root/miniconda3/etc/profile.d/conda.sh
conda activate safeear

export SAFEAR_CKPT=Exps/Ablation_S2_pe_learnable/checkpoints/epoch=7-val_eer=0.0247.ckpt
export SAFEAR_FEAT=wavlm

# Optional auth override
export SAFEAR_WEB_USER=admin
export SAFEAR_WEB_PASSWORD=safeear123
export SAFEAR_WEB_TOKEN=safeear-demo-token

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
- `POST /api/auth/login` body: `{ "username": "...", "password": "..." }`
- `POST /api/predict` form-data: `file`, optional `threshold`, `max_len`
- `POST /api/predict_batch` form-data: `files[]`, optional `threshold`, `max_len`
- `GET /api/tasks/{job_id}`
- `GET /api/history?limit=100`
- `GET /api/history/export.csv?limit=100`
- `GET /api/history/export.json?limit=100`
- `GET /api/audit?limit=100`
