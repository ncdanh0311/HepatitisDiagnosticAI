"""
src/data/processor.py
=====================
Pure data-processing logic — zero UI dependencies.
Used by both the Streamlit app and the legacy Tkinter UI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import IterativeImputer


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "hepatitis.csv"

# ── Category definitions ─────────────────────────────────────────────────────

CATEGORY_RAW_LABELS = {
    "0=Blood Donor":          0,
    "0s=suspect Blood Donor": 1,
    "1=Hepatitis":            2,
    "2=Fibrosis":             3,
    "3=Cirrhosis":            4,
}

CATEGORY_NAMES = [
    "Blood Donor",
    "Suspect Blood Donor",
    "Hepatitis",
    "Fibrosis",
    "Cirrhosis",
]

BINARY_MAP = {
    "0=Blood Donor":          0,
    "0s=suspect Blood Donor": 0,
    "1=Hepatitis":            1,
    "2=Fibrosis":             1,
    "3=Cirrhosis":            1,
}

BINARY_NAMES = ["Healthy", "Liver Disease"]

FEATURE_COLUMNS = ["Age", "Sex", "ALB", "ALP", "ALT", "AST",
                   "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]

FEATURE_DESCRIPTIONS = {
    "ALB":  "Albumin — protein do gan sản xuất",
    "ALP":  "Alkaline Phosphatase — enzyme gan-mật",
    "ALT":  "Alanine Aminotransferase — chỉ thị tổn thương gan",
    "AST":  "Aspartate Aminotransferase — tổn thương tế bào",
    "BIL":  "Bilirubin — sắc tố mật",
    "CHE":  "Cholinesterase — khả năng tổng hợp gan",
    "CHOL": "Cholesterol — mỡ máu",
    "CREA": "Creatinine — chức năng thận",
    "GGT":  "Gamma-Glutamyl Transferase — tổn thương gan-mật",
    "PROT": "Total Proteins — tổng protein huyết thanh",
    "Age":  "Tuổi bệnh nhân",
    "Sex":  "Giới tính (0=Nam, 1=Nữ)",
}


class DataProcessor:
    """
    Đọc và xử lý dữ liệu hepatitis.csv.

    Attributes:
        data:           DataFrame đang làm việc
        raw_data:       Bản gốc chưa sửa đổi
        zscore:         Ma trận Z-Score đã tính
        feature_list:   Danh sách các cột đặc trưng đã trích lọc
    """

    def __init__(self, filepath: str | Path | None = None) -> None:
        self.filepath = Path(filepath) if filepath else DATA_FILE
        self.raw_data: pd.DataFrame | None = None
        self.data: pd.DataFrame = self._read_file()
        self.zscore: pd.DataFrame | None = None
        self.zscore_threshold: float | None = None
        self.n_components: int | None = None
        self.dropped_columns: list[str] = []
        self.selected_columns: list[str] = []
        self.feature_list: list = []

    # ── I/O ─────────────────────────────────────────────────────────────────

    def _read_file(self) -> pd.DataFrame:
        """Đọc CSV, mã hóa Sex, Category thành số."""
        df = pd.read_csv(self.filepath)
        self.raw_data = df.copy()
        df["Sex"] = df["Sex"].map({"m": 0, "f": 1})
        df["Category"] = df["Category"].map(CATEGORY_RAW_LABELS)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df

    def reload(self) -> None:
        """Reset về trạng thái ban đầu."""
        self.data = self._read_file()
        self.zscore = None
        self.zscore_threshold = None
        self.n_components = None
        self.dropped_columns.clear()
        self.selected_columns.clear()
        self.feature_list.clear()

    # ── Inspection ──────────────────────────────────────────────────────────

    def missing_summary(self) -> pd.DataFrame:
        """Trả về bảng thống kê missing values."""
        total = self.data.isnull().sum()
        pct = total / len(self.data) * 100
        return pd.DataFrame({"Missing": total, "Percent (%)": pct.round(2)})\
                 .query("Missing > 0")\
                 .sort_values("Missing", ascending=False)

    def describe(self) -> pd.DataFrame:
        return self.data.describe().round(3)

    def value_counts_category(self) -> pd.Series:
        """Phân bố nhãn Category."""
        return self.data["Category"].value_counts().sort_index()

    def correlation_matrix(self) -> pd.DataFrame:
        return self.data.select_dtypes(include=np.number).corr().round(3)

    # ── Cleaning ────────────────────────────────────────────────────────────

    def fill_missing_mode(self) -> pd.DataFrame:
        self.data = self.data.fillna(self.data.mode().iloc[0])
        return self.data

    def fill_missing_iterative(self, max_iter: int = 20) -> pd.DataFrame:
        """MICE imputation — dự đoán giá trị thiếu."""
        imputer = IterativeImputer(max_iter=max_iter, random_state=42,
                                   initial_strategy="median")
        num_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        self.data[num_cols] = imputer.fit_transform(self.data[num_cols])
        return self.data

    def drop_null_rows(self) -> pd.DataFrame:
        self.data = self.data.dropna(how="any")
        return self.data

    def drop_columns(self, columns: list[str]) -> pd.DataFrame:
        self.data = self.data.drop(
            columns=[c for c in columns if c in self.data.columns], axis=1
        )
        return self.data

    # ── Transforms ──────────────────────────────────────────────────────────

    def minmax_scale(self) -> pd.DataFrame:
        scaler = preprocessing.MinMaxScaler()
        num = self.data.select_dtypes(include=np.number)
        self.data[num.columns] = scaler.fit_transform(num)
        return self.data

    def compute_zscore(self) -> pd.DataFrame:
        num = self.data.select_dtypes(include=np.number)
        self.zscore = pd.DataFrame(
            np.abs(stats.zscore(num, nan_policy="omit")),
            columns=num.columns,
            index=self.data.index,
        )
        return self.zscore

    def remove_outliers(self, threshold: float) -> tuple[pd.DataFrame, int]:
        """Xóa outliers dựa trên Z-Score. Trả về (data_cleaned, n_removed)."""
        if self.zscore is None:
            self.compute_zscore()
        before = len(self.data)
        mask = (self.zscore < threshold).all(axis=1)
        self.data = self.data[mask].reset_index(drop=True)
        self.zscore_threshold = threshold
        return self.data, before - len(self.data)

    # ── Feature Extraction ──────────────────────────────────────────────────

    def extract_pca(
        self, input_cols: list[str], output_col: str, n_components: int
    ) -> dict:
        x = self.data[input_cols]
        y = self.data[[output_col]]
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x)
        top_features = [
            x.columns[np.argmax(abs(pca.components_[i]))]
            for i in range(n_components)
        ]
        self.feature_list.append(top_features)
        return {
            "x_transformed": x_pca,
            "y": y,
            "top_features": top_features,
            "variance_ratio": pca.explained_variance_ratio_,
        }

    def extract_kbest(
        self, input_cols: list[str], output_col: str, k: int
    ) -> dict:
        x = self.data[input_cols]
        y = self.data[[output_col]]
        selector = SelectKBest(chi2, k=k)
        selector.fit(x, y)
        x_sel = selector.transform(x)
        selected = x.columns[selector.get_support(indices=True)].tolist()
        self.feature_list.append(selected)
        return {"x_selected": x_sel, "y": y, "selected_columns": selected}

    # ── ML-ready split ───────────────────────────────────────────────────────

    def get_ml_data(self, binary: bool = True) -> tuple:
        """
        Trả về (X_imputed, y) sẵn sàng cho training.
        Áp dụng IterativeImputer trên bản copy, không ảnh hưởng self.data.
        """
        df = self._read_file()
        if binary:
            y_map = BINARY_MAP
        else:
            y_map = CATEGORY_RAW_LABELS

        df["Category"] = pd.read_csv(self.filepath)["Category"].map(y_map)
        df["Sex"] = df["Sex"].map({0: 0, 1: 1}).fillna(0)  # already encoded

        # Re-encode Sex from raw if needed
        raw = pd.read_csv(self.filepath)
        df["Sex"] = raw["Sex"].map({"m": 0, "f": 1})
        df["Category"] = raw["Category"].map(y_map)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        y = df["Category"].values
        X = df.drop(columns=["Category"])

        imputer = IterativeImputer(max_iter=20, random_state=42,
                                   initial_strategy="median")
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        target_names = BINARY_NAMES if binary else CATEGORY_NAMES
        return X_imp, y, target_names
