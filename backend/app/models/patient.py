"""
backend/app/models/patient.py
==============================
Patient + MedicalRecord ORM models.
"""
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(10), nullable=True)
    contact_phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    records: Mapped[list["MedicalRecord"]] = relationship("MedicalRecord", back_populates="patient")
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="patient")


class MedicalRecord(Base):
    __tablename__ = "medical_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), index=True
    )
    recorded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    # Biomarkers (hepatitis blood panel)
    age: Mapped[float | None] = mapped_column(Float)
    sex: Mapped[float | None] = mapped_column(Float)   # 0=Female, 1=Male
    alb: Mapped[float | None] = mapped_column(Float)
    alp: Mapped[float | None] = mapped_column(Float)
    alt: Mapped[float | None] = mapped_column(Float)
    ast: Mapped[float | None] = mapped_column(Float)
    bil: Mapped[float | None] = mapped_column(Float)
    che: Mapped[float | None] = mapped_column(Float)
    chol: Mapped[float | None] = mapped_column(Float)
    crea: Mapped[float | None] = mapped_column(Float)
    ggt: Mapped[float | None] = mapped_column(Float)
    prot: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    patient: Mapped[Patient] = relationship("Patient", back_populates="records")
