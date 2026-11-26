
import json
import logging
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, ConfigDict


# JSON logging configuration
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        if hasattr(record, "method"):
            log_data["method"] = record.method
            log_data["path"] = record.path
            log_data["status_code"] = record.status_code
            log_data["duration_ms"] = record.duration_ms

        return json.dumps(log_data)


logger = logging.getLogger("mp267")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.propagate = False


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "battery_start_pct",
    "session_minutes",
    "charger_kw",
    "temp_c",
    "vehicle_age_yr",
    "queue_before",
    "vehicle_car",
    "vehicle_three_wheeler",
    "vehicle_two_wheeler",
    "station_type_Highway",
    "station_type_Mall",
    "station_type_Residential",
]

app = FastAPI(title="MP267 EV Energy Prediction API")


# Log every HTTP request in JSON format
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000, 2
        )

        logger.info(
            "API request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    battery_start_pct: float
    session_minutes: float
    charger_kw: float = Field(ge=1.5)
    temp_c: float
    vehicle_age_yr: float
    queue_before: int = Field(ge=0, le=20)

    vehicle: Literal[
        "car", "two_wheeler", "three_wheeler"
    ]

    station_type: Literal[
        "Mall", "Highway", "Residential"
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = pd.DataFrame([request.model_dump()])

    input_data = pd.get_dummies(
        input_data,
        columns=["vehicle", "station_type"],
        dtype=int
    )

    input_data = input_data.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    return {"prediction": float(prediction)}
