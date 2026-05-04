"""pages/3_🩺_Prediction.py — Patient prediction"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import CONFIG
from src.data.processor import DataProcessor, FEATURE_DESCRIPTIONS
from src.models.trainer import ModelTrainer

st.set_page_config(page_title="Prediction · Hepatitis", page_icon="🩺", layout="wide")
st.title("🩺 Patient Prediction")

# ── Model: prefer saved multiclass, else train ─────────────────────────────────
@st.cache_resource(show_spinner="Đang tải model...")
def get_model() -> ModelTrainer:
    path = CONFIG.model_path(binary=False)
    if path.exists():
        return ModelTrainer.load(path)
    proc = DataProcessor()
    X, y, names = proc.get_ml_data(binary=False)
    trainer = ModelTrainer(X, y, names, test_size=CONFIG.test_size)
    trainer.train()
    trainer.save(path)
    return trainer

trainer = get_model()
st.info(
    f"🏆 **{trainer.best_model_name}** · "
    f"Test accuracy: **{trainer.best_result.test_accuracy:.2%}** · "
    "5-class prediction"
)

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📝 Nhập chỉ số bệnh nhân")

FIELDS = [
    ("Age",  "Tuổi",         1,    80,    40,   1  ),
    ("ALB",  "Albumin",      10.0, 60.0,  41.6, 0.1),
    ("ALP",  "ALP",          10.0, 500.0, 68.0, 0.1),
    ("ALT",  "ALT",          1.0,  2000.0,25.0, 0.1),
    ("AST",  "AST",          1.0,  500.0, 34.0, 0.1),
    ("BIL",  "Bilirubin",    1.0,  250.0, 11.0, 0.1),
    ("CHE",  "Cholinesterase",1.0, 20.0,   8.0, 0.1),
    ("CHOL", "Cholesterol",   1.0, 10.0,   5.8, 0.01),
    ("CREA", "Creatinine",    1.0, 2000.0,73.0, 0.1),
    ("GGT",  "GGT",          1.0, 700.0,  29.0, 0.1),
    ("PROT", "Total Proteins",30.0,100.0, 71.0, 0.1),
]

c1, c2 = st.columns(2)
patient: dict = {}
for i, (key, label, lo, hi, default, step) in enumerate(FIELDS):
    col = c1 if i % 2 == 0 else c2
    patient[key] = float(col.number_input(
        f"{label} ({key})", float(lo), float(hi), float(default), float(step),
        help=FEATURE_DESCRIPTIONS.get(key, ""),
    ))

sex = st.radio("Giới tính", ["Nam", "Nữ"], horizontal=True)
patient["Sex"] = 0.0 if sex == "Nam" else 1.0

# Align to training feature order
feature_order = list(trainer.X.columns)
patient_df = pd.DataFrame([[patient.get(f, 0.0) for f in feature_order]],
                           columns=feature_order)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Dự đoán", type="primary"):
    pred_idx, confidence = trainer.predict_new(patient_df)
    pred_name = trainer.target_names[pred_idx]

    SEVERITY = {
        "Blood Donor":         ("🟢", "success", "Khỏe mạnh"),
        "Suspect Blood Donor": ("🟡", "warning", "Cần theo dõi thêm"),
        "Hepatitis":           ("🟠", "warning", "Viêm gan — cần điều trị"),
        "Fibrosis":            ("🔴", "error",   "Xơ hóa gan — cần can thiệp"),
        "Cirrhosis":           ("🔴", "error",   "Xơ gan — giai đoạn nghiêm trọng"),
    }
    icon, level, advice = SEVERITY.get(pred_name, ("❓", "info", ""))

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Chẩn đoán", f"{icon} {pred_name}")
    c2.metric("Confidence", f"{confidence:.1%}")
    c3.metric("Mô hình", trainer.best_model_name)
    getattr(st, level)(advice)

    # Probability chart
    if hasattr(trainer.best_result.model, "predict_proba"):
        proba = trainer.best_result.model.predict_proba(patient_df)[0]
        fig = go.Figure(go.Bar(
            x=trainer.target_names, y=proba,
            marker=dict(color=proba,
                        colorscale=[[0,"#1a1f2e"],[1,"#00D4FF"]]),
            text=[f"{p:.1%}" for p in proba], textposition="outside",
        ))
        fig.update_layout(
            title="Xác suất từng giai đoạn", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0,1.15], gridcolor="rgba(255,255,255,.08)"),
            font=dict(color="#E8EAF0"), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
