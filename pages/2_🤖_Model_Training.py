"""pages/2_🤖_Model_Training.py — ML Training Pipeline"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import CONFIG
from src.data.processor import DataProcessor
from src.models.trainer import ModelTrainer

st.set_page_config(page_title="Training · Hepatitis", page_icon="🤖", layout="wide")
st.title("🤖 Model Training")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")
    binary = st.radio("Chế độ",
        ["Nhị phân (Healthy / Disease)", "Đa lớp (5 giai đoạn)"]) \
        == "Nhị phân (Healthy / Disease)"
    test_size = st.slider("Test size (%)", 10, 40, 20) / 100

    retrain = st.button("🔄 Xoá cache & train lại")

# ── Load or train ─────────────────────────────────────────────────────────────
model_path = CONFIG.model_path(binary)

def _do_train(binary: bool, test_size: float) -> ModelTrainer:
    proc = DataProcessor()
    X, y, target_names = proc.get_ml_data(binary=binary)
    trainer = ModelTrainer(X, y, target_names, test_size=test_size)
    trainer.train()
    trainer.save(model_path)
    return trainer

if retrain and model_path.exists():
    model_path.unlink()
    st.cache_resource.clear()

@st.cache_resource
def get_trainer(binary: bool, test_size: float) -> ModelTrainer:
    if model_path.exists():
        try:
            return ModelTrainer.load(model_path)
        except Exception:
            pass
    return _do_train(binary, test_size)

col_btn, col_info = st.columns([1, 3])
run = col_btn.button("▶ Huấn luyện", type="primary", use_container_width=True)
col_info.info(
    f"Chế độ: **{'Nhị phân' if binary else 'Đa lớp'}** · "
    f"Test: **{int(test_size*100)}%** · "
    f"Cache: {'✅ Có sẵn' if model_path.exists() else '❌ Chưa có'}"
)

if run:
    with st.spinner("Đang huấn luyện..."):
        trainer = _do_train(binary, test_size)
        st.cache_resource.clear()
    st.success("✅ Xong!")
else:
    with st.spinner("Đang tải model..."):
        trainer = get_trainer(binary, test_size)

results = trainer.results
names   = list(results.keys())
COLORS  = ["#00D4FF", "#0072FF", "#8B5CF6"]

# ── Accuracy comparison ───────────────────────────────────────────────────────
st.subheader("📊 So sánh Accuracy")
fig = go.Figure()
fig.add_trace(go.Bar(
    name="CV Mean", x=names,
    y=[results[n].cv_mean for n in names],
    error_y=dict(type="data", array=[results[n].cv_std for n in names], visible=True),
    marker_color=COLORS, opacity=0.9,
    text=[f"{results[n].cv_mean:.3f}" for n in names], textposition="outside",
))
fig.add_trace(go.Scatter(
    name="Test Acc", x=names,
    y=[results[n].test_accuracy for n in names],
    mode="markers+text",
    marker=dict(size=14, color="white", line=dict(width=2, color=COLORS[0])),
    text=[f"{results[n].test_accuracy:.3f}" for n in names],
    textposition="top center",
))
fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)",
                  yaxis=dict(range=[0,1.1], gridcolor="rgba(255,255,255,.08)"),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02),
                  font=dict(color="#E8EAF0"))
st.plotly_chart(fig, use_container_width=True)

# ── Per-model tabs ────────────────────────────────────────────────────────────
st.subheader("📋 Chi tiết")
tabs = st.tabs([f"{'🏆 ' if n==trainer.best_model_name else ''}{n}" for n in names])

for tab, name in zip(tabs, names):
    r = results[name]
    with tab:
        c1, c2, c3 = st.columns(3)
        c1.metric("CV Accuracy", f"{r.cv_mean:.4f}", f"±{r.cv_std:.4f}")
        c2.metric("Test Accuracy", f"{r.test_accuracy:.4f}")
        c3.metric("Best", "✅" if name == trainer.best_model_name else "—")

        col_cm, col_rep = st.columns([1, 1.4])
        with col_cm:
            fig_cm = px.imshow(r.confusion_matrix,
                x=trainer.target_names, y=trainer.target_names,
                text_auto=True, color_continuous_scale="Blues", aspect="auto",
                labels=dict(x="Predicted", y="Actual"))
            fig_cm.update_layout(height=270, paper_bgcolor="rgba(0,0,0,0)",
                                 coloraxis_showscale=False, font=dict(color="#E8EAF0"))
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_rep:
            st.code(r.report, language="")

# ── Feature importance ────────────────────────────────────────────────────────
if "Random Forest" in results and not results["Random Forest"].feature_importances.empty:
    st.subheader("🔍 Feature Importance")
    fi = results["Random Forest"].feature_importances
    fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues",
                    text=fi["Importance"].map("{:.4f}".format))
    fig_fi.update_layout(height=370, paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)",
                         yaxis=dict(autorange="reversed"),
                         coloraxis_showscale=False, font=dict(color="#E8EAF0"))
    st.plotly_chart(fig_fi, use_container_width=True)
