"""
Module Huấn luyện Mô hình (Model Training Pipeline)
====================================================
Dự đoán cột Category (giai đoạn bệnh viêm gan) dựa trên các chỉ số sinh hóa.

Kỹ thuật sử dụng:
- IterativeImputer: Xử lý giá trị NA thông minh (dự đoán giá trị thiếu)
- SMOTE: Xử lý mất cân bằng dữ liệu (oversampling lớp thiểu số)
- 3 thuật toán: Random Forest, XGBoost, SVM
- Cross-Validation 5-fold + Classification Report + Confusion Matrix

"""

import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


def configure_console_output():
    """Make console output safer on Windows terminals with legacy encodings."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except ValueError:
                pass


configure_console_output()

import matplotlib

matplotlib.use("Agg")

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARNING] XGBoost chưa được cài đặt. Sẽ bỏ qua model XGBoost.")
    print("         Cài đặt: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("[WARNING] imbalanced-learn chưa được cài đặt. Sẽ dùng class_weight thay thế.")
    print("         Cài đặt: pip install imbalanced-learn")

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "hepatitis.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADER & PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class DataPreprocessor:
    """
    Đọc và tiền xử lý dữ liệu hepatitis.csv.

    Bước xử lý:
    1. Đọc CSV, chuyển đổi cột Sex (m/f → 0/1)
    2. Mã hóa Category thành nhãn số (multi-class: 4 lớp)
    3. Xử lý missing data bằng IterativeImputer
    4. Chuẩn hóa features bằng StandardScaler
    """

    # Mapping gốc cho Category (giữ nguyên 4 giai đoạn bệnh)
    CATEGORY_MAP = {
        "0=Blood Donor": 0,
        "0s=suspect Blood Donor": 1,
        "1=Hepatitis": 2,
        "2=Fibrosis": 3,
        "3=Cirrhosis": 4,
    }

    CATEGORY_NAMES = [
        "Blood Donor",
        "Suspect Blood Donor",
        "Hepatitis",
        "Fibrosis",
        "Cirrhosis",
    ]

    # Mapping nhị phân (Khỏe mạnh vs Bệnh)
    BINARY_MAP = {
        "0=Blood Donor": 0,
        "0s=suspect Blood Donor": 0,
        "1=Hepatitis": 1,
        "2=Fibrosis": 1,
        "3=Cirrhosis": 1,
    }

    BINARY_NAMES = ["Healthy", "Liver Disease"]

    def __init__(self, filepath: str = None, binary: bool = True):
        """
        Args:
            filepath: Đường dẫn tới file CSV. Mặc định: hepatitis.csv
            binary:   True = phân loại nhị phân (Khỏe mạnh/Bệnh),
                      False = phân loại 5 giai đoạn
        """
        self.filepath = filepath or str(DATA_FILE)
        self.binary = binary
        self.raw_data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None

    def load_and_process(self) -> tuple:
        """
        Đọc dữ liệu, tiền xử lý và trả về (X, y).

        Returns:
            (X, y): Features đã impute + scale, Labels đã encode
        """
        print("=" * 60)
        print("📊 BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU")
        print("=" * 60)

        # Đọc CSV
        df = pd.read_csv(self.filepath)
        self.raw_data = df.copy()
        print(f"  ✅ Đọc thành công: {df.shape[0]} dòng × {df.shape[1]} cột")

        # Hiển thị missing data TRƯỚC khi xử lý
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            print(f"\n  ⚠️  Missing values phát hiện:")
            for col, count in missing_cols.items():
                pct = count / len(df) * 100
                print(f"     {col}: {count} ({pct:.1f}%)")
        else:
            print("  ✅ Không có missing values")

        # Mã hóa Sex
        df["Sex"] = df["Sex"].map({"m": 0, "f": 1})
        print(f"\n  ✅ Mã hóa cột Sex: m→0, f→1")

        # Mã hóa Category (target)
        if self.binary:
            df["Category"] = df["Category"].map(self.BINARY_MAP)
            self.target_names = self.BINARY_NAMES
            print(f"  ✅ Mã hóa Category (nhị phân): Healthy=0, Disease=1")
        else:
            df["Category"] = df["Category"].map(self.CATEGORY_MAP)
            self.target_names = self.CATEGORY_NAMES
            print(f"  ✅ Mã hóa Category (5 lớp)")

        # Xóa cột Unnamed: 0
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Tách X, y
        self.y = df["Category"].values
        self.X = df.drop(columns=["Category"])
        self.feature_names = list(self.X.columns)

        # ── IterativeImputer ─────────────────────────────────────
        print(f"\n{'─' * 60}")
        print("🔧 BƯỚC 2: XỬ LÝ MISSING DATA (IterativeImputer)")
        print(f"{'─' * 60}")
        print("  Sử dụng IterativeImputer (MICE algorithm)")
        print("  → Dự đoán giá trị thiếu dựa trên các cột khác")

        imputer = IterativeImputer(
            max_iter=20,
            random_state=42,
            initial_strategy="median",
        )
        self.X = pd.DataFrame(
            imputer.fit_transform(self.X),
            columns=self.feature_names,
        )

        remaining_na = self.X.isnull().sum().sum()
        print(f"  ✅ Hoàn tất! Còn lại {remaining_na} missing values")

        # Phân bố target
        print(f"\n  📈 Phân bố nhãn (trước SMOTE):")
        unique, counts = np.unique(self.y, return_counts=True)
        for label, count in zip(unique, counts):
            name = self.target_names[int(label)]
            pct = count / len(self.y) * 100
            bar = "█" * int(pct / 2)
            print(f"     [{int(label)}] {name}: {count} ({pct:.1f}%) {bar}")

        return self.X, self.y


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Huấn luyện và đánh giá 3 mô hình ML:
    - Random Forest (Ensemble)
    - XGBoost (Gradient Boosting)
    - SVM (Support Vector Machine)

    Kỹ thuật:
    - SMOTE (nếu có) hoặc class_weight='balanced'
    - StandardScaler
    - Stratified K-Fold Cross-Validation
    """

    def __init__(self, X, y, target_names: list, test_size: float = 0.2):
        self.X = X
        self.y = y
        self.target_names = target_names
        self.test_size = test_size
        self.cv_splits = 5
        self.results = {}
        self.best_model = None
        self.best_model_name = None

        # Train/Test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"\n{'=' * 60}")
        print("📊 BƯỚC 3: CHIA DỮ LIỆU TRAIN/TEST")
        print(f"{'=' * 60}")
        print(f"  Train: {len(self.X_train)} mẫu ({(1-test_size)*100:.0f}%)")
        print(f"  Test:  {len(self.X_test)} mẫu ({test_size*100:.0f}%)")

    def _build_models(self) -> dict:
        """Tạo dictionary các mô hình cần huấn luyện."""
        models = {}
        smote_sampler = self._build_smote_sampler()

        # ── Random Forest ────────────────────────────────────────
        if HAS_IMBLEARN and smote_sampler is not None:
            models["Random Forest"] = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", smote_sampler),
                ("clf", RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                )),
            ])
        else:
            models["Random Forest"] = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                )),
            ])

        # ── XGBoost ──────────────────────────────────────────────
        if HAS_XGBOOST:
            # Tính scale_pos_weight cho imbalanced data
            n_classes = len(np.unique(self.y))
            if n_classes == 2:
                neg = np.sum(self.y == 0)
                pos = np.sum(self.y == 1)
                scale_weight = neg / pos if pos > 0 else 1
            else:
                scale_weight = 1

            xgb_params = dict(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            if n_classes == 2:
                xgb_params["scale_pos_weight"] = scale_weight

            if HAS_IMBLEARN and smote_sampler is not None:
                models["XGBoost"] = ImbPipeline([
                    ("scaler", StandardScaler()),
                    ("smote", smote_sampler),
                    ("clf", XGBClassifier(**xgb_params)),
                ])
            else:
                models["XGBoost"] = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", XGBClassifier(**xgb_params)),
                ])

        # ── SVM ──────────────────────────────────────────────────
        if HAS_IMBLEARN and smote_sampler is not None:
            models["SVM (RBF)"] = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", smote_sampler),
                ("clf", SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                )),
            ])
        else:
            models["SVM (RBF)"] = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                )),
            ])

        return models

    def _build_smote_sampler(self):
        """Choose a SMOTE configuration that is safe for the smallest class."""
        if not HAS_IMBLEARN:
            return None

        _, counts = np.unique(self.y_train, return_counts=True)
        min_class_count = int(counts.min())
        min_train_fold_count = int(np.floor(min_class_count * (self.cv_splits - 1) / self.cv_splits))
        if min_train_fold_count <= 1:
            return None

        k_neighbors = min(5, min_train_fold_count - 1)
        if k_neighbors < 1:
            return None

        return SMOTE(random_state=42, k_neighbors=k_neighbors)

    def train_and_evaluate(self) -> dict:
        """
        Huấn luyện tất cả model, đánh giá bằng cross-validation và test set.

        Returns:
            dict: Kết quả cho mỗi model (accuracy, cv_scores, report, etc.)
        """
        models = self._build_models()
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=42)

        print(f"\n{'=' * 60}")
        print("🚀 BƯỚC 4: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH")
        print(f"{'=' * 60}")

        smote_sampler = self._build_smote_sampler()
        if HAS_IMBLEARN and smote_sampler is not None:
            smote_status = f"SMOTE(k_neighbors={smote_sampler.k_neighbors})"
        else:
            smote_status = "class_weight='balanced'"
        print(f"  Xử lý imbalanced data: {smote_status}")
        print(f"  Cross-validation: Stratified {self.cv_splits}-Fold")
        print()

        best_score = 0

        for name, model in models.items():
            print(f"  {'─' * 50}")
            print(f"  🔹 {name}")
            print(f"  {'─' * 50}")

            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring="accuracy")
            print(f"    CV Accuracy:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"    CV Scores:    {[f'{s:.4f}' for s in cv_scores]}")

            # Fit trên toàn bộ train set
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            test_acc = np.mean(y_pred == self.y_test)
            print(f"    Test Accuracy: {test_acc:.4f}")

            # Classification Report
            report = classification_report(
                self.y_test, y_pred,
                target_names=self.target_names,
                zero_division=0,
            )
            print(f"\n    Classification Report:")
            for line in report.split("\n"):
                print(f"    {line}")

            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)

            self.results[name] = {
                "model": model,
                "cv_scores": cv_scores,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "test_accuracy": test_acc,
                "y_pred": y_pred,
                "confusion_matrix": cm,
                "report": report,
            }

            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name

        print(f"\n{'=' * 60}")
        print(f"🏆 MÔ HÌNH TỐT NHẤT: {self.best_model_name}")
        print(f"   CV Accuracy: {best_score:.4f}")
        print(f"{'=' * 60}")

        return self.results

    def plot_comparison(self, save_path: str = None):
        """
        Vẽ biểu đồ so sánh các mô hình.

        Args:
            save_path: Nếu cung cấp, lưu hình ảnh ra file. Nếu không, hiển thị.
        """
        if not self.results:
            print("⚠️ Chưa có kết quả. Hãy chạy train_and_evaluate() trước.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("📊 So sánh Hiệu suất Các Mô hình", fontsize=16, fontweight="bold")

        # ── 1. Bar chart: Test Accuracy ──────────────────────────
        names = list(self.results.keys())
        test_accs = [self.results[n]["test_accuracy"] for n in names]
        cv_means = [self.results[n]["cv_mean"] for n in names]
        cv_stds = [self.results[n]["cv_std"] for n in names]

        colors = ["#3498DB", "#E74C3C", "#2ECC71"][:len(names)]
        x = np.arange(len(names))

        axes[0].bar(x - 0.15, cv_means, 0.3, label="CV Mean", color=colors, alpha=0.8,
                    yerr=cv_stds, capsize=5)
        axes[0].bar(x + 0.15, test_accs, 0.3, label="Test Acc", color=colors, alpha=0.5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, fontsize=9)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy Comparison")
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
        for i, (cv, ta) in enumerate(zip(cv_means, test_accs)):
            axes[0].text(i - 0.15, cv + 0.02, f"{cv:.3f}", ha="center", fontsize=8)
            axes[0].text(i + 0.15, ta + 0.02, f"{ta:.3f}", ha="center", fontsize=8)

        # ── 2. Box plot: CV Scores Distribution ──────────────────
        cv_data = [self.results[n]["cv_scores"] for n in names]
        bp = axes[1].boxplot(cv_data, labels=names, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Cross-Validation Score Distribution")

        # ── 3. Confusion Matrix (best model) ─────────────────────
        best = self.results[self.best_model_name]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=best["confusion_matrix"],
            display_labels=self.target_names,
        )
        disp.plot(ax=axes[2], cmap="Blues", colorbar=False)
        axes[2].set_title(f"Confusion Matrix: {self.best_model_name}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n📁 Biểu đồ đã lưu: {save_path}")
        else:
            plt.show()

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Trích xuất Feature Importance từ Random Forest (nếu có).

        Returns:
            DataFrame chứa tên feature và mức độ quan trọng.
        """
        if "Random Forest" not in self.results:
            print("⚠️ Cần huấn luyện Random Forest trước.")
            return pd.DataFrame()

        rf_pipeline = self.results["Random Forest"]["model"]
        # Lấy classifier từ pipeline
        clf = rf_pipeline.named_steps["clf"]
        importances = clf.feature_importances_

        df_imp = pd.DataFrame({
            "Feature": self.X.columns if hasattr(self.X, "columns") else [f"f{i}" for i in range(len(importances))],
            "Importance": importances,
        }).sort_values("Importance", ascending=False)

        print(f"\n{'=' * 40}")
        print("📊 FEATURE IMPORTANCE (Random Forest)")
        print(f"{'=' * 40}")
        for _, row in df_imp.iterrows():
            bar = "█" * int(row["Importance"] * 50)
            print(f"  {row['Feature']:>6}: {row['Importance']:.4f} {bar}")

        return df_imp


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_training_pipeline(binary: bool = True, save_plot: bool = False):
    """
    Chạy toàn bộ pipeline huấn luyện.

    Args:
        binary:    True = Khỏe mạnh vs Bệnh, False = 5 lớp chi tiết
        save_plot: True = lưu biểu đồ ra file, False = hiển thị matplotlib
    """
    print("\n" + "═" * 60)
    print("  🏥 HEPATITIS DISEASE PREDICTION PIPELINE")
    print("  📋 Dự đoán giai đoạn bệnh viêm gan từ chỉ số sinh hóa")
    print("═" * 60)

    mode = "Nhị phân (Healthy / Disease)" if binary else "Đa lớp (5 giai đoạn)"
    print(f"\n  📌 Chế độ phân loại: {mode}\n")

    # 1. Tiền xử lý dữ liệu
    preprocessor = DataPreprocessor(binary=binary)
    X, y = preprocessor.load_and_process()

    # 2. Huấn luyện & đánh giá
    trainer = ModelTrainer(X, y, target_names=preprocessor.target_names)
    results = trainer.train_and_evaluate()

    # 3. Feature Importance
    trainer.get_feature_importance()

    # 4. Biểu đồ so sánh
    plot_path = str(BASE_DIR / "model_comparison.png") if save_plot else None
    trainer.plot_comparison(save_path=plot_path)

    return trainer


if __name__ == "__main__":
    # Chạy pipeline nhị phân (phổ biến nhất)
    trainer = run_training_pipeline(binary=True, save_plot=True)

    print("\n\n" + "═" * 60)
    print("  🔄 Chạy thêm chế độ ĐA LỚP (5 giai đoạn)...")
    print("═" * 60)

    # Chạy thêm pipeline đa lớp
    trainer_multi = run_training_pipeline(binary=False, save_plot=True)
