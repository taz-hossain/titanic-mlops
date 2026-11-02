"""
Serve the model as a REST API with FastAPI.
"""
import csv
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from features import engineer_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
PREDICTION_LOG = ROOT / "logs" / "predictions.csv"
API_KEY = os.getenv("API_KEY", "dev-secret-key")
MAX_BATCH = 100

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predicts whether a Titanic passenger survived. Send passenger details, get a prediction.",
    version="1.0.0",
)

_model = None
_model_info = {}


def get_model():
    global _model, _model_info
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not found. Run `python src/train.py` first.")
        _model = joblib.load(MODEL_PATH)
        if METRICS_PATH.exists():
            m = json.load(open(METRICS_PATH))
            _model_info = {"model_name": m.get("best_model"), "model_version": m.get("model_version")}
    return _model


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: Optional[str] = Security(api_key_header)):
    # compare_digest avoids timing attacks
    if key is None or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")
    return key


class Passenger(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "Pclass": 1, "Name": "Cumings, Mrs. John Bradley", "Sex": "female", "Age": 38,
        "SibSp": 1, "Parch": 0, "Fare": 71.28, "Cabin": "C85", "Embarked": "C"}})

    Pclass: Literal[1, 2, 3] = Field(..., description="Ticket class: 1, 2 or 3")
    Name: str = Field("Unknown, Mr. Passenger", max_length=200,
                      description="Format 'Surname, Title. First' (title is used as a feature)")
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(None, ge=0, le=100, description="Leave empty if unknown")
    SibSp: int = Field(0, ge=0, le=10)
    Parch: int = Field(0, ge=0, le=10)
    Fare: Optional[float] = Field(None, ge=0, le=1000)
    Cabin: Optional[str] = Field(None, max_length=50)
    Embarked: Optional[Literal["C", "Q", "S"]] = None


class Prediction(BaseModel):
    survived: int
    survival_probability: float
    label: str
    model_name: Optional[str] = None
    model_version: Optional[int] = None


def _predict(passengers: List[Passenger]) -> List[Prediction]:
    model = get_model()
    raw = pd.DataFrame([p.model_dump() for p in passengers])
    X = engineer_features(raw)
    probas = model.predict_proba(X)[:, 1]
    results = [
        Prediction(survived=int(pr >= 0.5), survival_probability=round(float(pr), 4),
                   label="Survived" if pr >= 0.5 else "Did not survive", **_model_info)
        for pr in probas
    ]
    _log_predictions(raw, results)
    return results


def _log_predictions(raw: pd.DataFrame, results: List[Prediction]):
    """Append every request + prediction to a CSV. monitor.py uses this as 'current' production data."""
    PREDICTION_LOG.parent.mkdir(exist_ok=True)
    new_file = not PREDICTION_LOG.exists()
    ts = datetime.now(timezone.utc).isoformat()
    with open(PREDICTION_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", *raw.columns, "prediction", "prediction_proba"])
        for row, r in zip(raw.itertuples(index=False), results):
            w.writerow([ts, *row, r.survived, r.survival_probability])


@app.get("/", tags=["info"])
def root():
    return {"message": "Titanic Survival API. See /docs for usage."}


@app.get("/health", tags=["info"])
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model-info", tags=["info"], dependencies=[Depends(verify_api_key)])
def model_info():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "No metrics found")
    return json.load(open(METRICS_PATH))


@app.post("/predict", response_model=Prediction, tags=["prediction"],
          dependencies=[Depends(verify_api_key)])
def predict(passenger: Passenger):
    return _predict([passenger])[0]


@app.post("/predict/batch", response_model=List[Prediction], tags=["prediction"],
          dependencies=[Depends(verify_api_key)])
def predict_batch(passengers: List[Passenger]):
    if not passengers:
        raise HTTPException(400, "Send at least one passenger")
    if len(passengers) > MAX_BATCH:
        raise HTTPException(413, f"Max {MAX_BATCH} passengers per request")
    return _predict(passengers)