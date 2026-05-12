"""
backend/app/api/v1/endpoints/patients.py
=========================================
Patient CRUD + medical record endpoints.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user_id, require_doctor
from app.db.session import get_db
from app.models.patient import MedicalRecord, Patient
from app.schemas.patient import (
    MedicalRecordCreate,
    MedicalRecordResponse,
    PatientCreate,
    PatientResponse,
    PatientUpdate,
)

router = APIRouter(prefix="/patients", tags=["patients"])


def _generate_code(seq: int) -> str:
    return f"HEP-{seq:06d}"


@router.get("", response_model=dict)
async def list_patients(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: str = Query(""),
    payload: dict = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    q = select(Patient)
    if search:
        q = q.where(
            Patient.full_name.ilike(f"%{search}%")
            | Patient.patient_code.ilike(f"%{search}%")
        )
    total = await db.scalar(select(func.count()).select_from(q.subquery()))
    rows = (await db.scalars(q.offset((page - 1) * size).limit(size))).all()
    return {
        "total": total,
        "page": page,
        "size": size,
        "items": [PatientResponse.model_validate(p) for p in rows],
    }


@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    body: PatientCreate,
    payload: dict = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    count = await db.scalar(select(func.count()).select_from(Patient))
    patient = Patient(
        patient_code=_generate_code((count or 0) + 1),
        created_by=uuid.UUID(payload["sub"]),
        **body.model_dump(),
    )
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return PatientResponse.model_validate(patient)


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: uuid.UUID,
    payload: dict = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    patient = await db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PatientResponse.model_validate(patient)


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: uuid.UUID,
    body: PatientUpdate,
    payload: dict = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    patient = await db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(patient, k, v)
    await db.commit()
    await db.refresh(patient)
    return PatientResponse.model_validate(patient)


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: uuid.UUID,
    payload: dict = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    patient = await db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    await db.delete(patient)
    await db.commit()


@router.get("/{patient_id}/records", response_model=list[MedicalRecordResponse])
async def get_records(
    patient_id: uuid.UUID,
    payload: dict = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.scalars(
            select(MedicalRecord).where(MedicalRecord.patient_id == patient_id)
        )
    ).all()
    return [MedicalRecordResponse.model_validate(r) for r in rows]


@router.post(
    "/{patient_id}/records",
    response_model=MedicalRecordResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_record(
    patient_id: uuid.UUID,
    body: MedicalRecordCreate,
    payload: dict = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    patient = await db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    record = MedicalRecord(
        patient_id=patient_id,
        recorded_by=uuid.UUID(payload["sub"]),
        **body.model_dump(),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return MedicalRecordResponse.model_validate(record)
