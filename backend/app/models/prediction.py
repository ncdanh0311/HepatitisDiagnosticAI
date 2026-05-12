"""
backend/app/models/prediction.py
=================================
Prediction + ModelVersion ORM models.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(255))
    accuracy: Mapped[float | None] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    artifact_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="model_version")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )
    medical_record_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("medical_records.id"), nullable=True
    )
    model_version_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("model_versions.id"), nullable=True
    )
    predicted_class: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    shap_values: Mapped[dict | None] = mapped_column(JSON)
    input_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    performed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    patient: Mapped["Patient"] = relationship("Patient", back_populates="predictions")  # type: ignore
    model_version: Mapped[ModelVersion | None] = relationship("ModelVersion", back_populates="predictions")
