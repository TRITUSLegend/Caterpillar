# ML Service API Contracts

Request/response shapes for the ML service, consumed by the Node backend.
Shapes marked **Approved** are frozen — any change is announced before it ships.

- Base URL (local): `http://localhost:8000`
- All bodies are JSON, field names are `snake_case`.
- Invalid input (missing field, wrong type, value outside an allowed set) returns FastAPI's standard **422** response — never a silent default.
- Extra fields in a request body are ignored, so a whole MongoDB document (with `_id` etc.) can be forwarded as long as the required fields are present.

| Endpoint | Status |
|---|---|
| `GET /health` | **Approved** |
| `POST /predict/task-time` | **Approved** |
| `POST /predict/anomaly` | **Approved** |
| `POST /predict/maintenance-score` | **Approved** |
| `POST /generate-handover-report` | TBD (Phase 4) |

---

## GET /health — Approved

Liveness check. Returns once both models are loaded.

**Response**
```json
{"status": "ok", "models_loaded": ["anomaly", "task_time"]}
```

---

## POST /predict/task-time — Approved

Predicts how long a task will actually take. Takes one Task document.

**Request**
```json
{
  "task_id": "T0001",
  "task_type": "Excavation",
  "weather": "Rainy",
  "site_terrain": "Muddy",
  "operator_skill": "Beginner",
  "machine_age": 3.2,
  "estimated_time": 80
}
```

| Field | Type | Source (Task) | Allowed values |
|---|---|---|---|
| `task_id` | string, optional | `Task_ID` | echoed back only |
| `task_type` | string | `Task_Type` | `Excavation`, `Trenching`, `Loading`, `Hauling`, `Grading` |
| `weather` | string | `Weather` | `Clear`, `Cloudy`, `Rainy`, `Windy` |
| `site_terrain` | string | `Site_Terrain` | `Flat`, `Rocky`, `Muddy`, `Sloped` |
| `operator_skill` | string | `Operator_Skill` | `Beginner`, `Intermediate`, `Expert` |
| `machine_age` | number | `Machine_Age` | years, >= 0 |
| `estimated_time` | integer | `Estimated_Time` | minutes, > 0 |

**Response**
```json
{
  "task_id": "T0001",
  "predicted_time": 131.4,
  "estimated_time": 80,
  "predicted_overrun": 51.4
}
```
- `predicted_time`: minutes, 1 decimal.
- `predicted_overrun`: `predicted_time - estimated_time` in minutes (negative = expected to finish early).
- `task_id`: echoed if sent, otherwise `null`.

---

## POST /predict/anomaly — Approved

Flags a single telemetry reading as normal or anomalous. Takes one Telemetry document.

**Request**
```json
{
  "machine_id": "M03",
  "machine_type": "Excavator",
  "timestamp": "2026-09-16T07:00:00Z",
  "hydraulic_pressure": 298.4,
  "fuel_used": 37.1,
  "idling_time": 12,
  "load_cycles": 48
}
```

| Field | Type | Source (Telemetry) | Allowed values |
|---|---|---|---|
| `machine_id` | string | `Machine_ID` | echoed back |
| `machine_type` | string | `Machine_Type` | `Excavator`, `Wheel Loader`, `Dozer`, `Motor Grader`, `Articulated Truck` |
| `timestamp` | string | `Timestamp` | ISO 8601 UTC, echoed back |
| `hydraulic_pressure` | number | `Hydraulic_Pressure` | bar |
| `fuel_used` | number | `Fuel_Used` | litres, >= 0 |
| `idling_time` | integer | `Idling_Time` | minutes, 0–120 |
| `load_cycles` | integer | `Load_Cycles` | >= 0 |

**Response**
```json
{
  "machine_id": "M03",
  "timestamp": "2026-09-16T07:00:00Z",
  "is_anomaly": true,
  "anomaly_score": 0.087
}
```
- `is_anomaly`: boolean — the flag the backend should act on.
- `anomaly_score`: relative score, higher = more unusual. `is_anomaly` is `true` exactly when `anomaly_score > 0`. Not a probability.
- The 0 boundary is deliberate: `anomaly_score` is the negated Isolation Forest `decision_function`, which is offset so that 0 sits at the model's 3% contamination threshold — about 3% of training readings score above 0. On live data the flagged rate follows the incoming readings (a misbehaving machine will exceed 3%), but the cutoff stays fixed at 0.

---

## POST /predict/maintenance-score — Approved

Rule-based (no trained model) maintenance urgency for one machine.

> **Different from the other endpoints:** this needs a **time range** of telemetry for one machine, not a single document. Query Telemetry by `Machine_ID` for the last ~3 days (~24 readings, minimum 2) and send them all.

**Request**
```json
{
  "machine_id": "M03",
  "readings": [
    {
      "timestamp": "2026-09-16T07:00:00Z",
      "hydraulic_pressure": 298.4,
      "fuel_used": 37.1,
      "idling_time": 12,
      "hours_since_service": 452.3
    }
  ]
}
```

| Field | Source (Telemetry) |
|---|---|
| `timestamp` | `Timestamp` |
| `hydraulic_pressure` | `Hydraulic_Pressure` |
| `fuel_used` | `Fuel_Used` |
| `idling_time` | `Idling_Time` |
| `hours_since_service` | `Hours_Since_Service` |

`readings` must contain at least 2 items.

**Response**
```json
{
  "machine_id": "M03",
  "score": 74,
  "status": "Red",
  "reasons": [
    "Hydraulic pressure down 16.6 bar across window",
    "493 h since last service (interval 500 h)"
  ],
  "readings_used": 24
}
```
- `score`: integer 0–100, higher = more urgent maintenance.
- `status`: `"Green"` (< 40), `"Amber"` (40–69), `"Red"` (>= 70). Computed by the ML service — the backend passes it through and does not apply its own thresholds.
- `reasons`: human-readable strings, biggest contributor first; empty when `Green`.

Score is the sum of three rule-based components:

| Component | Max points |
|---|---|
| Hydraulic pressure below rated 330 bar or falling across window | 40 |
| Hours since service vs 500 h interval | 35 |
| Fuel usage upward trend | 25 |

---

## POST /generate-handover-report — TBD (Phase 4)

Generates a natural-language shift handover summary via the Gemini API. Shape to be defined in Phase 4.

Known constraints:
- Task has no `Shift_ID` or timestamp, so Task and Telemetry records are joined on `Operator_ID` + `Machine_ID` only.
- If Gemini is unavailable (timeout, rate limit), the endpoint returns an error response rather than crashing the service.
