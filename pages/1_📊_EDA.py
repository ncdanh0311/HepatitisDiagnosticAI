"""pages/1_📊_EDA.py — Exploratory Data Analysis"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.express as px
import streamlit as st

from config import CONFIG
from src.data.processor import DataProcessor

st.set_page_config(page_title="EDA · Hepatitis", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

@st.cache_resource
def get_processor() -> DataProcessor:
    return DataProcessor()

proc = get_processor()

CATEGORY_NAMES = {0:"Blood Donor",1:"Suspect BD",2:"Hepatitis",3:"Fibrosis",4:"Cirrhosis"}
df = proc.data.copy()
df["Category Label"] = df["Category"].map(CATEGORY_NAMES)
num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "Category"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Hiển thị")
    show_null = st.checkbox("Missing values", True)
    show_dist = st.checkbox("Phân bố chỉ số",  True)
    show_corr = st.checkbox("Tương quan",       True)
    show_box  = st.checkbox("Box plot",         False)

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("📋 Dữ liệu", expanded=True):
    st.dataframe(df.drop(columns=["Category Label"]), use_container_width=True, height=300)
    c1, c2, c3 = st.columns(3)
    c1.metric("Dòng", len(df))
    c2.metric("Cột", len(df.columns) - 1)
    c3.metric("Duplicates", int(df.duplicated().sum()))

# ── Missing values ────────────────────────────────────────────────────────────
if show_null:
    st.subheader("⚠️ Missing Values")
    miss_df = proc.missing_summary()
    if miss_df.empty:
        st.success("Không có missing values!")
    else:
        c1, c2 = st.columns([1, 2])
        c1.dataframe(miss_df, use_container_width=True)
        fig = px.bar(miss_df.reset_index(), x="index", y="Missing",
                     text="Percent (%)", color="Missing",
                     color_continuous_scale="Blues",
                     labels={"index": "Cột"})
        fig.update_layout(showlegend=False, height=240,
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          coloraxis_showscale=False)
        c2.plotly_chart(fig, use_container_width=True)

# ── Distribution ──────────────────────────────────────────────────────────────
if show_dist:
    st.subheader("📈 Phân bố chỉ số sinh hóa")
    selected = st.multiselect("Chọn cột", num_cols,
                               default=["ALT", "AST", "BIL", "GGT", "ALB"])
    if selected:
        fig = px.violin(
            df.melt(value_vars=selected, var_name="Chỉ số", value_name="Giá trị"),
            x="Chỉ số", y="Giá trị", color="Chỉ số",
            box=True, points="outliers",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, height=370,
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # Category bar
    cat = df["Category Label"].value_counts().reset_index()
    cat.columns = ["Category", "Count"]
    fig2 = px.bar(cat, x="Category", y="Count", color="Category",
                  text="Count", color_discrete_sequence=px.colors.qualitative.Bold)
    fig2.update_layout(showlegend=False, height=280,
                       paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

# ── Correlation ───────────────────────────────────────────────────────────────
if show_corr:
    st.subheader("🔥 Ma trận tương quan")
    fig = px.imshow(proc.correlation_matrix(), text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# ── Box plot ──────────────────────────────────────────────────────────────────
if show_box:
    st.subheader("📦 Box plot theo Category")
    feat = st.selectbox("Chỉ số", num_cols,
                        index=num_cols.index("ALT") if "ALT" in num_cols else 0)
    fig = px.box(df, x="Category Label", y=feat, color="Category Label",
                 points="outliers", notched=True,
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(showlegend=False, height=380,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 Thống kê mô tả"):
    st.dataframe(proc.describe(), use_container_width=True)
