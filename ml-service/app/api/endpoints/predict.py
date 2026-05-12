"""
ml-service/app/api/endpoints/predict.py
=========================================
POST /predict — run inference + optional SHAP explanation.
"""
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.predictor import LABEL_MAP, Predictor
from app.services.shap_explainer import SHAPExplainer

router = APIRouter()
_explainer = SHAPExplainer()


class PredictRequest(BaseModel):
    features: dict[str, Any]
    explain: bool = False


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    shap: dict | None = None


@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    try:
        predictor = Predictor.get()
        result = predictor.predict(body.features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    shap_data = None
    if body.explain:
        # Map predicted class string back to index
        pred_idx = next(
            (k for k, v in LABEL_MAP.items() if v == result["predicted_class"]), 0
        )
        shap_data = _explainer.explain(body.features, pred_idx)

    return PredictResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        shap=shap_data,
    )


@router.get("/health")
async def health():
    return {"status": "ok", "model_loaded": Predictor._instance is not None}
