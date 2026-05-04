"""
config.py — Centralized project configuration
"""
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # Paths
    data_path:   Path = field(default_factory=lambda: BASE_DIR / "data" / "hepatitis.csv")
    model_dir:   Path = field(default_factory=lambda: BASE_DIR / "models")

    # Model training
    test_size:   float = 0.2
    cv_splits:   int   = 5
    random_seed: int   = 42

    # NLP
    nlp_confidence_threshold: float = 0.12

    def __post_init__(self):
        self.model_dir.mkdir(exist_ok=True)

    def model_path(self, binary: bool) -> Path:
        tag = "binary" if binary else "multiclass"
        return self.model_dir / f"trainer_{tag}.pkl"


CONFIG = Config()
