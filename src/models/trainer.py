"""
src/models/trainer.py
=====================
Model training pipeline: Random Forest · XGBoost · SVM
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


@dataclass
class ModelResult:
    """Lưu kết quả của một mô hình."""
    name: str
    model: Any
    cv_scores: np.ndarray
    test_accuracy: float
    y_pred: np.ndarray
    confusion_matrix: np.ndarray
    report: str
    feature_importances: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def cv_mean(self) -> float:
        return float(self.cv_scores.mean())

    @property
    def cv_std(self) -> float:
        return float(self.cv_scores.std())


class ModelTrainer:
    """
    Huấn luyện & đánh giá 3 thuật toán ML:
    - Random Forest (Ensemble)
    - XGBoost (Gradient Boosting)
    - SVM với kernel RBF

    Kỹ thuật xử lý imbalanced data:
    - SMOTE (nếu imbalanced-learn đã cài đặt)
    - class_weight='balanced' (fallback)
    """

    CV_SPLITS = 5

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        target_names: list[str],
        test_size: float = 0.2,
    ) -> None:
        self.X = X
        self.y = y
        self.target_names = target_names
        self.test_size = test_size
        self.results: dict[str, ModelResult] = {}
        self.best_model_name: str | None = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

    # ── SMOTE helper ─────────────────────────────────────────────────────────

    def _safe_smote(self):
        """Tạo SMOTE với k_neighbors an toàn cho mọi CV fold."""
        if not HAS_IMBLEARN:
            return None
        _, counts = np.unique(self.y_train, return_counts=True)
        min_fold = int(np.floor(counts.min() * (self.CV_SPLITS - 1) / self.CV_SPLITS))
        if min_fold <= 1:
            return None
        k = min(5, min_fold - 1)
        return SMOTE(random_state=42, k_neighbors=k)

    def _make_pipeline(self, clf, smote=None) -> Pipeline:
        steps = [("scaler", StandardScaler())]
        if smote is not None:
            return ImbPipeline(steps + [("smote", smote), ("clf", clf)])
        return Pipeline(steps + [("clf", clf)])

    # ── Build models ─────────────────────────────────────────────────────────

    def _build_models(self) -> dict[str, Pipeline]:
        smote = self._safe_smote()
        n_classes = len(np.unique(self.y))
        models: dict[str, Pipeline] = {}

        # Random Forest
        models["Random Forest"] = self._make_pipeline(
            RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
            smote,
        )

        # XGBoost
        if HAS_XGBOOST:
            xgb_params: dict[str, Any] = dict(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42,
            )
            if n_classes == 2:
                neg = int(np.sum(self.y == 0))
                pos = int(np.sum(self.y == 1))
                xgb_params["scale_pos_weight"] = neg / pos if pos else 1
            models["XGBoost"] = self._make_pipeline(
                XGBClassifier(**xgb_params), smote
            )

        # SVM
        models["SVM (RBF)"] = self._make_pipeline(
            SVC(kernel="rbf", C=10, gamma="scale",
                class_weight="balanced", probability=True, random_state=42),
            smote,
        )

        return models

    # ── Train ────────────────────────────────────────────────────────────────

    def train(self) -> dict[str, ModelResult]:
        """Huấn luyện tất cả model, trả về dict kết quả."""
        models = self._build_models()
        cv = StratifiedKFold(n_splits=self.CV_SPLITS, shuffle=True, random_state=42)
        best_cv = 0.0

        for name, pipeline in models.items():
            cv_scores = cross_val_score(
                pipeline, self.X_train, self.y_train, cv=cv, scoring="accuracy"
            )
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            test_acc = float(np.mean(y_pred == self.y_test))

            report = classification_report(
                self.y_test, y_pred,
                target_names=self.target_names,
                zero_division=0,
            )
            cm = confusion_matrix(self.y_test, y_pred)

            # Feature importance (RF only)
            fi_df = pd.DataFrame()
            if name == "Random Forest":
                fi_df = pd.DataFrame({
                    "Feature": self.X.columns,
                    "Importance": pipeline.named_steps["clf"].feature_importances_,
                }).sort_values("Importance", ascending=False)

            self.results[name] = ModelResult(
                name=name,
                model=pipeline,
                cv_scores=cv_scores,
                test_accuracy=test_acc,
                y_pred=y_pred,
                confusion_matrix=cm,
                report=report,
                feature_importances=fi_df,
            )

            if cv_scores.mean() > best_cv:
                best_cv = cv_scores.mean()
                self.best_model_name = name

        return self.results

    @property
    def best_result(self) -> ModelResult | None:
        if self.best_model_name:
            return self.results[self.best_model_name]
        return None

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Lưu toàn bộ trainer ra file .pkl."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ModelTrainer":
        """Tải trainer đã lưu từ file .pkl."""
        return joblib.load(path)

    def predict_new(self, patient_df: pd.DataFrame) -> tuple[int, float]:
        """
        Dự đoán ca bệnh mới.

        Returns:
            (predicted_class_index, confidence)
        """
        if not self.best_model_name:
            raise RuntimeError("Chưa huấn luyện model. Hãy gọi train() trước.")
        model = self.results[self.best_model_name].model
        pred_class = int(model.predict(patient_df)[0])
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(patient_df)[0][pred_class])
        else:
            confidence = 1.0
        return pred_class, confidence
