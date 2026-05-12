"""
backend/app/schemas/patient.py
================================
Pydantic v2 schemas for patient & medical record CRUD.
"""
import uuid
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Patient ──────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=255)
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    contact_phone: Optional[str] = None


class PatientUpdate(BaseModel):
    full_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    contact_phone: Optional[str] = None


class PatientResponse(BaseModel):
    id: uuid.UUID
    patient_code: str
    full_name: str
    date_of_birth: Optional[date]
    gender: Optional[str]
    contact_phone: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Medical Record ────────────────────────────────────────────────────────────

class MedicalRecordCreate(BaseModel):
    age: Optional[float] = None
    sex: Optional[float] = None
    alb: Optional[float] = None
    alp: Optional[float] = None
    alt: Optional[float] = None
    ast: Optional[float] = None
    bil: Optional[float] = None
    che: Optional[float] = None
    chol: Optional[float] = None
    crea: Optional[float] = None
    ggt: Optional[float] = None
    prot: Optional[float] = None
    notes: Optional[str] = None


class MedicalRecordResponse(MedicalRecordCreate):
    id: uuid.UUID
    patient_id: uuid.UUID
    recorded_at: datetime

    model_config = {"from_attributes": True}
