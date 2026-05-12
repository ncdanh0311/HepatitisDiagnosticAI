"""
backend/app/api/v1/endpoints/predictions.py
============================================
Prediction endpoint — calls ML service and persists result.
"""
import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.dependencies import get_current_user_id, require_doctor
from app.db.session import get_db
from app.models.prediction import ModelVersion, Prediction
from app.schemas.prediction import PredictionRequest, PredictionResponse, SHAPExplanation

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
async def create_prediction(
    body: PredictionRequest,
    payload: dict = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    # 1. Call ML service
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            ml_resp = await client.post(
                f"{settings.ML_SERVICE_URL}/predict",
                json={"features": body.features, "explain": True},
            )
            ml_resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"ML service error: {exc}")

    ml_data = ml_resp.json()

    # 2. Get active model version
    model_ver = await db.scalar(
        select(ModelVersion).where(ModelVersion.is_active == True)  # noqa
    )

    # 3. Parse SHAP
    shap_explanation = None
    if ml_data.get("shap"):
        shap_explanation = SHAPExplanation(**ml_data["shap"])

    # 4. Persist
    pred = Prediction(
        patient_id=body.patient_id,
        medical_record_id=body.medical_record_id,
        model_version_id=model_ver.id if model_ver else None,
        predicted_class=ml_data["predicted_class"],
        confidence=ml_data["confidence"],
        shap_values=ml_data.get("shap"),
        input_features=body.features,
        performed_by=uuid.UUID(payload["sub"]),
    )
    db.add(pred)
    await db.commit()
    await db.refresh(pred)

    return PredictionResponse(
        id=pred.id,
        patient_id=pred.patient_id,
        predicted_class=pred.predicted_class,
        confidence=pred.confidence,
        shap_explanation=shap_explanation,
        model_version=model_ver.version if model_ver else None,
        created_at=pred.created_at,
    )


@router.get("", response_model=list[PredictionResponse])
async def list_predictions(
    patient_id: uuid.UUID | None = Query(None),
    limit: int = Query(50, le=200),
    payload: dict = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    q = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
    if patient_id:
        q = q.where(Prediction.patient_id == patient_id)
    rows = (await db.scalars(q)).all()
    return [
        PredictionResponse(
            id=r.id,
            patient_id=r.patient_id,
            predicted_class=r.predicted_class,
            confidence=r.confidence,
            created_at=r.created_at,
        )
        for r in rows
    ]
