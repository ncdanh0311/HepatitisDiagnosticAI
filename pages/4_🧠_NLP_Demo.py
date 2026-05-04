"""pages/4_🧠_NLP_Demo.py — Intent Classifier Demo"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from config import CONFIG
from src.nlp.intent_classifier import IntentClassifier, INTENT_DESCRIPTIONS

st.set_page_config(page_title="NLP Demo · Hepatitis", page_icon="🧠", layout="wide")
st.title("🧠 NLP Intent Classifier")
st.caption("Gõ lệnh tiếng Việt — model phân loại intent và trả về chức năng tương ứng.")

@st.cache_resource(show_spinner="Đang huấn luyện NLP...")
def get_clf():
    clf = IntentClassifier(CONFIG.nlp_confidence_threshold)
    info = clf.train()
    return clf, info

clf, info = get_clf()

# ── Model info ────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Thông tin model", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Mẫu train", info["n_samples"])
    c2.metric("Số intent",  info["n_intents"])
    c3.metric("CV Accuracy", f"{info['cv_mean']:.2%}")
    st.caption("Pipeline: TF-IDF (char n-gram 2–4) → Logistic Regression")

# ── Quick examples ────────────────────────────────────────────────────────────
EXAMPLES = [
    "vẽ biểu đồ đường", "kiểm tra null", "huấn luyện mô hình",
    "chuẩn hóa dữ liệu", "dự đoán bệnh nhân", "trích lọc pca",
    "xóa biểu đồ", "xem dữ liệu",
]
st.markdown("**💡 Thử nhanh:**")
cols = st.columns(4)
selected = None
for i, ex in enumerate(EXAMPLES):
    if cols[i % 4].button(ex, key=f"ex{i}"):
        selected = ex

# ── Input ─────────────────────────────────────────────────────────────────────
text = st.text_input("🗣️ Nhập lệnh", value=selected or "",
                     placeholder="ví dụ: vẽ biểu đồ đường...")

if text.strip():
    result = clf.predict(text)
    top5   = clf.top_k(text, k=5)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Intent",    result["intent"] if result["is_confident"] else "❓ unknown")
    c2.metric("Confidence", f"{result['confidence']:.1%}")
    c3.metric("Chức năng",  result["description"])

    if result["is_confident"]:
        st.success(f"✅ **{result['description']}**")
    else:
        st.warning("⚠️ Không chắc chắn — xem gợi ý bên dưới.")

    fig = go.Figure(go.Bar(
        x=[t["confidence"] for t in top5],
        y=[t["description"] for t in top5],
        orientation="h",
        marker=dict(color=[t["confidence"] for t in top5],
                    colorscale=[[0,"#1a1f2e"],[1,"#00D4FF"]]),
        text=[f"{t['confidence']:.1%}" for t in top5],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top-5 Probability", height=260,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0,1.15], gridcolor="rgba(255,255,255,.06)"),
        yaxis=dict(autorange="reversed"), font=dict(color="#E8EAF0"),
        showlegend=False, margin=dict(l=10,r=80,t=40,b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📖 Tất cả intent"):
    for intent, desc in INTENT_DESCRIPTIONS.items():
        st.markdown(f"- `{intent}` — {desc}")
