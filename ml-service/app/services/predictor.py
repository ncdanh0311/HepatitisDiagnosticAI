"""
ml-service/app/services/predictor.py
======================================
Loads the trained sklearn/XGBoost pipeline and runs inference.
Migrated from the original src/models/trainer.py ModelTrainer.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Label mapping aligned with the original dataset
LABEL_MAP = {
    0: "Blood Donor",
    1: "Suspect BD",
    2: "Hepatitis",
    3: "Fibrosis",
    4: "Cirrhosis",
}

FEATURE_COLS = ["Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]


class Predictor:
    """Singleton that loads the active model on first access."""

    _instance: "Predictor | None" = None

    def __init__(self, model_path: Path) -> None:
        self._trainer = joblib.load(model_path)
        self.feature_cols = FEATURE_COLS

    @classmethod
    def get(cls) -> "Predictor":
        if cls._instance is None:
            model_path = Path(os.getenv("MODEL_PATH", "models/trainer_binary.pkl"))
            cls._instance = cls(model_path)
        return cls._instance

    def predict(self, features: dict[str, Any]) -> dict:
        """
        Args:
            features: dict with keys matching FEATURE_COLS

        Returns:
            { predicted_class, confidence, probabilities }
        """
        df = pd.DataFrame([features], columns=self.feature_cols)

        trainer = self._trainer
        if not trainer.best_model_name:
            raise RuntimeError("Model not trained. Run training pipeline first.")

        pipeline = trainer.results[trainer.best_model_name].model
        pred_idx = int(pipeline.predict(df)[0])
        probas: np.ndarray = pipeline.predict_proba(df)[0]
        confidence = float(probas[pred_idx])

        return {
            "predicted_class": LABEL_MAP.get(pred_idx, str(pred_idx)),
            "confidence": confidence,
            "probabilities": {
                LABEL_MAP.get(i, str(i)): float(p)
                for i, p in enumerate(probas)
            },
        }
