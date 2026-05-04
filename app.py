"""app.py — Hepatitis EDA · Home"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import CONFIG

st.set_page_config(
    page_title="Hepatitis EDA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS (loaded once here, inherited by all pages via st.set_page_config) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.kpi {
    background: linear-gradient(135deg,#1a1f2e,#252b3b);
    border: 1px solid rgba(0,212,255,.2);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    text-align: center;
}
.kpi-val {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg,#00D4FF,#0072FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.kpi-lbl { color:#8892a4; font-size:.85rem; margin-top:.2rem; }
[data-testid="stSidebar"] { background:#11151f !important; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(CONFIG.data_path)

df = load_data()

CAT_LABEL = {
    "0=Blood Donor": "Blood Donor",
    "0s=suspect Blood Donor": "Suspect BD",
    "1=Hepatitis": "Hepatitis",
    "2=Fibrosis": "Fibrosis",
    "3=Cirrhosis": "Cirrhosis",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Hepatitis EDA")
    st.caption("Phân tích & dự đoán bệnh viêm gan")
    st.markdown("---")
    st.markdown("""
**Điều hướng**
- 📊 EDA — Phân tích dữ liệu
- 🤖 Model Training — Huấn luyện ML
- 🩺 Prediction — Dự đoán ca bệnh
- 🧠 NLP Demo — Thử intent classifier
""")
    st.markdown("---")
    st.caption("Đặng Nguyễn Quang Huy · 2024")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0D1117,#1a1f2e,#0D2137);
            border:1px solid rgba(0,212,255,.25);border-radius:18px;
            padding:2.2rem 2.8rem;margin-bottom:1.8rem">
  <div style="font-size:2.6rem;font-weight:700;
              background:linear-gradient(90deg,#00D4FF,#0072FF,#a855f7);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    🏥 Hepatitis Disease Analysis
  </div>
  <div style="color:#8892a4;margin-top:.5rem;font-size:1rem">
    Phân tích và dự đoán bệnh viêm gan từ chỉ số sinh hóa máu ·
    Machine Learning + NLP Assistant
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
disease_cats = ["1=Hepatitis", "2=Fibrosis", "3=Cirrhosis"]
kpis = [
    ("👤", len(df), "Bệnh nhân"),
    ("🧪", df.shape[1] - 2, "Chỉ số sinh hóa"),
    ("⚠️", int(df.isnull().sum().sum()), "Missing values"),
    ("🔴", int(df["Category"].isin(disease_cats).sum()), "Ca bệnh gan"),
]
cols = st.columns(4)
for col, (icon, val, lbl) in zip(cols, kpis):
    col.markdown(f"""
    <div class="kpi">
      <div style="font-size:1.6rem">{icon}</div>
      <div class="kpi-val">{val:,}</div>
      <div class="kpi-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Phân bố nhãn")
    cat = df["Category"].map(CAT_LABEL).value_counts().reset_index()
    cat.columns = ["Category", "Count"]
    fig = px.pie(cat, values="Count", names="Category", hole=0.42,
                 color_discrete_sequence=["#00D4FF","#0072FF","#F59E0B","#EF4444","#8B5CF6"])
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      height=290, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Missing values")
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values()
    if miss.empty:
        st.success("Không có missing values!")
    else:
        fig2 = go.Figure(go.Bar(
            x=miss.values, y=miss.index, orientation="h",
            marker_color="#00D4FF", opacity=0.85,
            text=[f"{v} ({v/len(df)*100:.1f}%)" for v in miss.values],
            textposition="outside",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=290, margin=dict(t=10,b=10,l=10,r=80),
            xaxis=dict(showgrid=False), font=dict(color="#E8EAF0"),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Dataset preview ───────────────────────────────────────────────────────────
with st.expander("📋 Xem dữ liệu gốc"):
    st.dataframe(df, use_container_width=True, height=300)
