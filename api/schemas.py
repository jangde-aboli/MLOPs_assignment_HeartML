"""Request/response models. Field ranges follow the UCI attribute docs."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "age": 57, "sex": 1, "cp": 4, "trestbps": 140, "chol": 241, "fbs": 0,
        "restecg": 0, "thalach": 123, "exang": 1, "oldpeak": 0.2, "slope": 2,
        "ca": 0, "thal": 7,
    }})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., gt=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., gt=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., gt=0, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., ge=-5, le=10, description="ST depression vs rest")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: int | None = Field(None, ge=0, le=3,
                           description="Major vessels coloured (0-3); null if unknown")
    thal: Literal[3, 6, 7] | None = Field(
        None, description="3 normal, 6 fixed defect, 7 reversible defect; null if unknown")


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=100)


class Prediction(BaseModel):
    prediction: int
    label: str
    probability_disease: float
    confidence: float
    risk_level: str


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    model_name: str


class PredictResponse(Prediction):
    model_name: str
    request_id: str
