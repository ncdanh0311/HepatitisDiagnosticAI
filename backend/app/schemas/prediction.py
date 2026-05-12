"""
backend/app/schemas/prediction.py
===================================
Pydantic v2 schemas for prediction requests and responses.
"""
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    patient_id: uuid.UUID
    medical_record_id: Optional[uuid.UUID] = None
    # Direct biomarker input (for one-shot form submission)
    features: dict[str, Any] = Field(
        ...,
        example={
            "Age": 45, "Sex": 1, "ALB": 41.6, "ALP": 49.3,
            "ALT": 17.5, "AST": 24.7, "BIL": 4.8, "CHE": 9.67,
            "CHOL": 5.74, "CREA": 72, "GGT": 28.1, "PROT": 70.3,
        },
    )


class SHAPExplanation(BaseModel):
    feature_names: list[str]
    shap_values: list[float]
    base_value: float
    predicted_class: str


class PredictionResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    predicted_class: str
    confidence: float
    shap_explanation: Optional[SHAPExplanation] = None
    model_version: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}
