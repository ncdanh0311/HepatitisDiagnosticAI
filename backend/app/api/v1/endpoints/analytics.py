"""
backend/app/api/v1/endpoints/analytics.py
==========================================
Analytics dashboard data endpoints.
"""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user_id
from app.db.session import get_db
from app.models.patient import Patient
from app.models.prediction import Prediction

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/dashboard")
async def dashboard_kpis(
    payload: dict = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    total_patients = await db.scalar(select(func.count()).select_from(Patient))
    total_predictions = await db.scalar(select(func.count()).select_from(Prediction))

    # Predictions in last 30 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    recent_preds = await db.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.created_at >= cutoff)
    )

    # High-risk count (hepatitis/fibrosis/cirrhosis)
    high_risk = await db.scalar(
        select(func.count()).select_from(Prediction).where(
            Prediction.predicted_class.in_(["Hepatitis", "Fibrosis", "Cirrhosis"])
        )
    )

    return {
        "total_patients": total_patients or 0,
        "total_predictions": total_predictions or 0,
        "recent_predictions_30d": recent_preds or 0,
        "high_risk_count": high_risk or 0,
        "risk_percentage": round((high_risk / total_predictions * 100) if total_predictions else 0, 1),
    }


@router.get("/disease-distribution")
async def disease_distribution(
    payload: dict = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            select(Prediction.predicted_class, func.count().label("count"))
            .group_by(Prediction.predicted_class)
        )
    ).all()
    return [{"category": r.predicted_class, "count": r.count} for r in rows]


@router.get("/prediction-trends")
async def prediction_trends(
    days: int = 30,
    payload: dict = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = (
        await db.execute(
            select(
                func.date_trunc("day", Prediction.created_at).label("day"),
                func.count().label("count"),
            )
            .where(Prediction.created_at >= cutoff)
            .group_by("day")
            .order_by("day")
        )
    ).all()
    return [{"date": str(r.day.date()), "count": r.count} for r in rows]
