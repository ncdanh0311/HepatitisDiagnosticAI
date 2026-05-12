"""
ml-service/app/services/shap_explainer.py
==========================================
SHAP waterfall explanation for hepatitis predictions.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from app.services.predictor import FEATURE_COLS, Predictor


class SHAPExplainer:
    _explainer = None

    @classmethod
    def _get_explainer(cls):
        if cls._explainer is None and HAS_SHAP:
            predictor = Predictor.get()
            pipeline = predictor._trainer.results[predictor._trainer.best_model_name].model
            # Use TreeExplainer for RF/XGBoost, KernelExplainer as fallback
            clf = pipeline.named_steps["clf"]
            try:
                cls._explainer = shap.TreeExplainer(clf)
            except Exception:
                X_bg = pd.DataFrame(
                    np.zeros((1, len(FEATURE_COLS))), columns=FEATURE_COLS
                )
                cls._explainer = shap.KernelExplainer(pipeline.predict_proba, X_bg)
        return cls._explainer

    def explain(self, features: dict[str, Any], pred_class_idx: int) -> dict | None:
        if not HAS_SHAP:
            return None
        explainer = self._get_explainer()
        if explainer is None:
            return None

        predictor = Predictor.get()
        pipeline = predictor._trainer.results[predictor._trainer.best_model_name].model
        scaler = pipeline.named_steps["scaler"]
        df = pd.DataFrame([features], columns=FEATURE_COLS)
        X_scaled = scaler.transform(df)

        shap_vals = explainer.shap_values(X_scaled)

        # For multi-output: pick the predicted class dimension
        if isinstance(shap_vals, list):
            vals = shap_vals[pred_class_idx][0]
        else:
            vals = shap_vals[0]

        return {
            "feature_names": FEATURE_COLS,
            "shap_values": [float(v) for v in vals],
            "base_value": float(
                explainer.expected_value[pred_class_idx]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            ),
            "predicted_class": pred_class_idx,
        }
