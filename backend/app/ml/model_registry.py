"""
Model Registry — saves and loads trained model artifacts.

All models are stored under app/ml/models/ as joblib files.
Metadata (training stats, feature names, version) stored alongside.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional
import joblib

logger = logging.getLogger("mirofish.ml.registry")

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ModelRegistry:

    @staticmethod
    def model_path(key: str) -> Path:
        return MODELS_DIR / f"{key}.joblib"

    @staticmethod
    def meta_path(key: str) -> Path:
        return MODELS_DIR / f"{key}_meta.json"

    @staticmethod
    def save(key: str, data: Any) -> None:
        path = ModelRegistry.model_path(key)
        joblib.dump(data, path, compress=3)
        logger.info(f"Saved model '{key}' to {path}")

        # Save metadata separately as JSON for quick inspection
        meta = {k: v for k, v in data.items()
                if isinstance(v, (str, int, float, bool, list, dict)) and k != "features"}
        try:
            meta_path = ModelRegistry.meta_path(key)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception:
            pass

    @staticmethod
    def load(key: str) -> Optional[Any]:
        path = ModelRegistry.model_path(key)
        if not path.exists():
            logger.warning(f"Model '{key}' not found at {path}")
            return None
        try:
            data = joblib.load(path)
            logger.info(f"Loaded model '{key}' from {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load model '{key}': {e}")
            return None

    @staticmethod
    def is_trained(key: str) -> bool:
        return ModelRegistry.model_path(key).exists()

    @staticmethod
    def list_models() -> list:
        return [p.stem for p in MODELS_DIR.glob("*.joblib")]

    @staticmethod
    def get_meta(key: str) -> Optional[dict]:
        path = ModelRegistry.meta_path(key)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
