"""
ML service for the Smart Operator Assistant.

Run from the ml-service directory:
    uvicorn app.main:app --port 8000
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

ML_SERVICE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ML_SERVICE_DIR / "models"

# The anomaly pipeline pickle references `preprocessing.PerTypeScaler`
sys.path.insert(0, str(ML_SERVICE_DIR))

from app.maintenance import score_machine  # noqa: E402
from app.schemas import (  # noqa: E402
    AnomalyRequest,
    AnomalyResponse,
    MaintenanceRequest,
    MaintenanceResponse,
    TaskTimeRequest,
    TaskTimeResponse,
)

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["task_time"] = joblib.load(MODELS_DIR / "task_time_model.pkl")
    models["anomaly"] = joblib.load(MODELS_DIR / "anomaly_model.pkl")
    yield
    models.clear()


app = FastAPI(title="Smart Operator Assistant - ML Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": sorted(models)}


@app.post("/predict/task-time", response_model=TaskTimeResponse)
def predict_task_time(req: TaskTimeRequest):
    row = pd.DataFrame([{
        "Task_Type": req.task_type,
        "Weather": req.weather,
        "Site_Terrain": req.site_terrain,
        "Operator_Skill": req.operator_skill,
        "Machine_Age": req.machine_age,
        "Estimated_Time": req.estimated_time,
    }])
    predicted = round(float(models["task_time"].predict(row)[0]), 1)
    return TaskTimeResponse(
        task_id=req.task_id,
        predicted_time=predicted,
        estimated_time=req.estimated_time,
        predicted_overrun=round(predicted - req.estimated_time, 1),
    )


@app.post("/predict/anomaly", response_model=AnomalyResponse)
def predict_anomaly(req: AnomalyRequest):
    row = pd.DataFrame([{
        "Machine_Type": req.machine_type,
        "Timestamp": req.timestamp,
        "Hydraulic_Pressure": req.hydraulic_pressure,
        "Fuel_Used": req.fuel_used,
        "Idling_Time": req.idling_time,
        "Load_Cycles": req.load_cycles,
    }])
    # Flag is derived from the returned score so `is_anomaly == (anomaly_score > 0)` always holds
    score = round(-float(models["anomaly"].decision_function(row)[0]), 4)
    return AnomalyResponse(
        machine_id=req.machine_id,
        timestamp=req.timestamp,
        is_anomaly=score > 0,
        anomaly_score=score,
    )


@app.post("/predict/maintenance-score", response_model=MaintenanceResponse)
def predict_maintenance_score(req: MaintenanceRequest):
    score, status, reasons = score_machine(req.readings)
    return MaintenanceResponse(
        machine_id=req.machine_id,
        score=score,
        status=status,
        reasons=reasons,
        readings_used=len(req.readings),
    )
