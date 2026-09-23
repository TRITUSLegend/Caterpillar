# ML Service

FastAPI service for the Smart Operator Assistant: task time prediction, telemetry
anomaly detection and a rule-based maintenance score. Request/response shapes are
in [CONTRACTS.md](CONTRACTS.md).

## Run locally

From this directory (`ml-service/`):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Check it is up:

```bash
curl http://localhost:8000/health
# {"status":"ok","models_loaded":["anomaly","task_time"]}
```

Interactive docs: http://localhost:8000/docs

## Layout

| Path | Purpose |
|---|---|
| `app/main.py` | FastAPI app, loads models at startup, endpoints |
| `app/schemas.py` | Request/response validation (pydantic) |
| `app/maintenance.py` | Rule-based maintenance score |
| `preprocessing.py` | Shared transformer used inside the anomaly model - must stay importable |
| `models/` | Trained model files (`.pkl`) |
| `training/train_models.py` | Retrains both models from `../data/*.csv` |

## Retraining

```bash
python training/train_models.py
```

Models must be loaded with the same scikit-learn version they were trained with
(pinned in `requirements.txt`).
