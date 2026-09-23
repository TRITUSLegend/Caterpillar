"""Request/response models. Shapes must match CONTRACTS.md exactly."""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

TaskType = Literal["Excavation", "Trenching", "Loading", "Hauling", "Grading"]
Weather = Literal["Clear", "Cloudy", "Rainy", "Windy"]
SiteTerrain = Literal["Flat", "Rocky", "Muddy", "Sloped"]
OperatorSkill = Literal["Beginner", "Intermediate", "Expert"]
MachineType = Literal["Excavator", "Wheel Loader", "Dozer", "Motor Grader", "Articulated Truck"]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _check_timestamp(value: str) -> str:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("timestamp must be ISO 8601, e.g. 2026-09-16T07:00:00Z")
    return value


# ---------- /predict/task-time ----------

class TaskTimeRequest(RequestModel):
    task_id: Optional[str] = None
    task_type: TaskType
    weather: Weather
    site_terrain: SiteTerrain
    operator_skill: OperatorSkill
    machine_age: float = Field(ge=0)
    estimated_time: int = Field(gt=0)


class TaskTimeResponse(BaseModel):
    task_id: Optional[str]
    predicted_time: float
    estimated_time: int
    predicted_overrun: float


# ---------- /predict/anomaly ----------

class AnomalyRequest(RequestModel):
    machine_id: str
    machine_type: MachineType
    timestamp: str
    hydraulic_pressure: float
    fuel_used: float = Field(ge=0)
    idling_time: int = Field(ge=0, le=120)
    load_cycles: int = Field(ge=0)

    _ts = field_validator("timestamp")(_check_timestamp)


class AnomalyResponse(BaseModel):
    machine_id: str
    timestamp: str
    is_anomaly: bool
    anomaly_score: float


# ---------- /predict/maintenance-score ----------

class MaintenanceReading(RequestModel):
    timestamp: str
    hydraulic_pressure: float
    fuel_used: float = Field(ge=0)
    idling_time: int = Field(ge=0, le=120)
    hours_since_service: float = Field(ge=0)

    _ts = field_validator("timestamp")(_check_timestamp)


class MaintenanceRequest(RequestModel):
    machine_id: str
    readings: List[MaintenanceReading] = Field(min_length=2)


class MaintenanceResponse(BaseModel):
    machine_id: str
    score: int
    status: Literal["Green", "Amber", "Red"]
    reasons: List[str]
    readings_used: int
