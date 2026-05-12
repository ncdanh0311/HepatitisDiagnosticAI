"""initial_schema

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-05-12 11:09:44.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'a1b2c3d4e5f6'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── roles ────────────────────────────────────────────────────────────────
    op.create_table(
        'roles',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(50), nullable=False, unique=True),
    )

    # ── users ────────────────────────────────────────────────────────────────
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=False),
        sa.Column('hashed_password', sa.Text(), nullable=False),
        sa.Column('role_id', sa.Integer(),
                  sa.ForeignKey('roles.id', ondelete='SET NULL'), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('idx_users_email', 'users', ['email'], unique=True)
    op.create_index('idx_users_role',  'users', ['role_id'])

    # ── patients ─────────────────────────────────────────────────────────────
    op.create_table(
        'patients',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('patient_code', sa.String(50), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=False),
        sa.Column('date_of_birth', sa.Date(), nullable=True),
        sa.Column('gender', sa.String(10), nullable=True),
        sa.Column('contact_phone', sa.String(20), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
    )
    op.create_index('idx_patients_code', 'patients', ['patient_code'], unique=True)

    # ── medical_records ───────────────────────────────────────────────────────
    op.create_table(
        'medical_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False),
        sa.Column('recorded_by', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id'), nullable=True),
        sa.Column('age',  sa.Float(), nullable=True),
        sa.Column('sex',  sa.Float(), nullable=True),
        sa.Column('alb',  sa.Float(), nullable=True),
        sa.Column('alp',  sa.Float(), nullable=True),
        sa.Column('alt',  sa.Float(), nullable=True),
        sa.Column('ast',  sa.Float(), nullable=True),
        sa.Column('bil',  sa.Float(), nullable=True),
        sa.Column('che',  sa.Float(), nullable=True),
        sa.Column('chol', sa.Float(), nullable=True),
        sa.Column('crea', sa.Float(), nullable=True),
        sa.Column('ggt',  sa.Float(), nullable=True),
        sa.Column('prot', sa.Float(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
    )
    op.create_index('idx_records_patient', 'medical_records', ['patient_id'])

    # ── model_versions ────────────────────────────────────────────────────────
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('mlflow_run_id', sa.String(255), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.false()),
        sa.Column('artifact_path', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
        sa.UniqueConstraint('model_name', 'version', name='uq_model_version'),
    )

    # ── predictions ───────────────────────────────────────────────────────────
    op.create_table(
        'predictions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False),
        sa.Column('medical_record_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('medical_records.id'), nullable=True),
        sa.Column('model_version_id', sa.Integer(),
                  sa.ForeignKey('model_versions.id'), nullable=True),
        sa.Column('predicted_class', sa.String(50), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('shap_values', postgresql.JSON(), nullable=True),
        sa.Column('input_features', postgresql.JSON(), nullable=False),
        sa.Column('performed_by', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
    )
    op.create_index('idx_predictions_patient', 'predictions', ['patient_id'])
    op.create_index('idx_predictions_created', 'predictions', ['created_at'])

    # ── ai_chat_logs ──────────────────────────────────────────────────────────
    op.create_table(
        'ai_chat_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id'), nullable=True),
        sa.Column('session_id', sa.String(100), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('sources', postgresql.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
    )
    op.create_index('idx_chat_session', 'ai_chat_logs', ['session_id'])

    # ── audit_logs ────────────────────────────────────────────────────────────
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id'), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource', sa.String(100), nullable=True),
        sa.Column('resource_id', sa.String(255), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
    )
    op.create_index('idx_audit_user',   'audit_logs', ['user_id'])
    op.create_index('idx_audit_action', 'audit_logs', ['action'])

    # Seed roles
    op.execute("INSERT INTO roles (name) VALUES ('admin'), ('doctor'), ('researcher') ON CONFLICT DO NOTHING")


def downgrade() -> None:
    op.drop_table('audit_logs')
    op.drop_table('ai_chat_logs')
    op.drop_table('predictions')
    op.drop_table('model_versions')
    op.drop_table('medical_records')
    op.drop_table('patients')
    op.drop_table('users')
    op.drop_table('roles')
