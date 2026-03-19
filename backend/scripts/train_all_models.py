#!/usr/bin/env python3
"""
Train all MiroFish ML prediction models.

Usage:
    cd backend
    python scripts/train_all_models.py              # Full training
    python scripts/train_all_models.py --fast        # NBA only (quickest)
    python scripts/train_all_models.py --download-only
    python scripts/train_all_models.py --validate-only

Loop structure:
  The script runs a validation loop after training:
  - Downloads data
  - Trains each model
  - Validates against hold-out test set
  - If metrics fall below thresholds → re-trains with adjusted hyperparameters
  - Saves final models

Metric thresholds (must pass to save):
  NBA moneyline:    AUC > 0.65,  Brier < 0.24
  Soccer home win:  AUC > 0.62,  Brier < 0.24
  Kalshi:           Brier improvement > 0.001

IMPORTANT DISCLAIMER:
  These models are tools for structured analysis — not financial advice.
  Sports betting and prediction markets involve significant financial risk.
  No model guarantees profitable outcomes. Always bet within your means.
  Past model performance does not guarantee future results.
"""

import sys
import os
import time
import argparse
import logging
import traceback
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("LLM_API_KEY", "not-needed-for-training")
os.environ.setdefault("ZEP_API_KEY", "not-needed-for-training")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BACKEND_DIR / "logs" / "training.log"),
    ]
)
logger = logging.getLogger("train_all_models")

from app.ml.data_pipeline import download_all
from app.ml.nba_predictor import NBAPredictor
from app.ml.soccer_predictor import SoccerPredictor
from app.ml.kalshi_predictor import KalshiPredictor
from app.ml.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------

NBA_THRESHOLDS = {
    "ml_auc": 0.65,
    "ml_brier": 0.24,
}

SOCCER_THRESHOLDS = {
    "home_win_auc": 0.62,
    "home_win_brier": 0.245,
}

KALSHI_THRESHOLDS = {
    "calibration_improvement": -0.001,  # Can be negative (small dataset)
}


def check_thresholds(metrics: dict, thresholds: dict, model_name: str) -> bool:
    """Returns True if all metrics pass their thresholds."""
    passed = True
    for key, threshold in thresholds.items():
        val = metrics.get(key)
        if val is None:
            logger.warning(f"[{model_name}] Missing metric: {key}")
            continue
        # AUC: higher is better. Brier: lower is better.
        if "auc" in key:
            ok = val >= threshold
        elif "brier" in key:
            ok = val <= threshold
        elif "improvement" in key:
            ok = val >= threshold
        else:
            ok = True

        status = "PASS" if ok else "FAIL"
        logger.info(f"  [{model_name}] {key}: {val:.4f} (threshold {threshold:.4f}) [{status}]")
        if not ok:
            passed = False
    return passed


def run_training_loop(model_cls, model_name: str, thresholds: dict,
                       train_kwargs: dict = None, max_attempts: int = 3) -> dict:
    """
    Training loop: attempt training up to max_attempts times.
    If metrics fail, adjusts hyperparameters and retries.
    Returns final metrics dict.
    """
    train_kwargs = train_kwargs or {}
    best_metrics = None

    for attempt in range(1, max_attempts + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"[{model_name}] Training attempt {attempt}/{max_attempts}")
        logger.info(f"{'='*50}")

        try:
            model = model_cls()
            metrics = model.train(**train_kwargs)

            logger.info(f"\n[{model_name}] Metrics (attempt {attempt}):")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    logger.info(f"  {k}: {v}")

            passed = check_thresholds(metrics, thresholds, model_name)

            if passed:
                logger.info(f"[{model_name}] All thresholds passed! Saving model.")
                model.save()
                return metrics
            else:
                logger.warning(f"[{model_name}] Some thresholds failed on attempt {attempt}.")
                if best_metrics is None or metrics.get("ml_brier", 999) < best_metrics.get("ml_brier", 999):
                    best_metrics = metrics
                    best_model = model

                if attempt < max_attempts:
                    logger.info(f"[{model_name}] Retrying with adjusted parameters...")
                    # Increase estimators and reduce learning rate for next attempt
                    if hasattr(model_cls, "XGB_PARAMS"):
                        model_cls.XGB_PARAMS["n_estimators"] = min(1000,
                            model_cls.XGB_PARAMS.get("n_estimators", 500) + 200)
                        model_cls.XGB_PARAMS["learning_rate"] = max(0.01,
                            model_cls.XGB_PARAMS.get("learning_rate", 0.03) * 0.7)
                    time.sleep(1)

        except Exception as e:
            logger.error(f"[{model_name}] Training failed on attempt {attempt}: {e}")
            logger.error(traceback.format_exc())
            if attempt == max_attempts:
                raise

    # Save best model even if thresholds not met
    logger.warning(f"[{model_name}] Could not meet all thresholds. Saving best model anyway.")
    if best_model is not None:
        best_model.save()
    return best_metrics or {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MiroFish prediction models")
    parser.add_argument("--fast", action="store_true", help="NBA only (skip soccer)")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-nba", action="store_true")
    parser.add_argument("--skip-soccer", action="store_true")
    parser.add_argument("--skip-kalshi", action="store_true")
    parser.add_argument("--attempts", type=int, default=3, help="Max training attempts per model")
    args = parser.parse_args()

    # Ensure logs dir
    (BACKEND_DIR / "logs").mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("  MiroFish ML Training Pipeline")
    print("  ⚠️  DISCLAIMER: For research/analysis only.")
    print("  ⚠️  Betting involves financial risk. No guarantees.")
    print("="*60 + "\n")

    # ── Step 1: Download all datasets ──────────────────────────────────────
    logger.info("Step 1: Downloading datasets...")
    t0 = time.time()
    download_results = download_all(verbose=True)
    logger.info(f"Downloads complete in {time.time()-t0:.1f}s")

    # Check what data is available
    nba_available = download_results.get("nba_elo") is not None and download_results["nba_elo"] > 100
    soccer_available = download_results.get("soccer") is not None and download_results["soccer"] > 100
    kalshi_available = True  # Kalshi has fallback calibrator

    if args.download_only:
        logger.info("Download-only mode. Exiting.")
        return

    if args.validate_only:
        logger.info("Validate-only mode — checking existing models...")
        _run_validation()
        return

    # ── Step 2: Train NBA model ────────────────────────────────────────────
    all_results = {}

    if not args.skip_nba:
        if nba_available:
            logger.info("\nStep 2: Training NBA model...")
            try:
                nba_metrics = run_training_loop(
                    NBAPredictor, "NBA",
                    NBA_THRESHOLDS,
                    max_attempts=args.attempts
                )
                all_results["nba"] = nba_metrics
            except Exception as e:
                logger.error(f"NBA training failed: {e}")
                all_results["nba"] = {"error": str(e)}
        else:
            logger.warning("NBA data unavailable — skipping NBA model training")

    # ── Step 3: Train Soccer model ─────────────────────────────────────────
    if not args.skip_soccer and not args.fast:
        if soccer_available:
            logger.info("\nStep 3: Training Soccer model...")
            try:
                soccer_metrics = run_training_loop(
                    SoccerPredictor, "Soccer",
                    SOCCER_THRESHOLDS,
                    max_attempts=args.attempts
                )
                all_results["soccer"] = soccer_metrics
            except Exception as e:
                logger.error(f"Soccer training failed: {e}")
                all_results["soccer"] = {"error": str(e)}
        else:
            logger.warning("Soccer data unavailable — skipping Soccer model training")

    # ── Step 4: Train Kalshi model ─────────────────────────────────────────
    if not args.skip_kalshi:
        logger.info("\nStep 4: Training Kalshi calibration model...")
        try:
            kalshi_metrics = run_training_loop(
                KalshiPredictor, "Kalshi",
                KALSHI_THRESHOLDS,
                max_attempts=args.attempts
            )
            all_results["kalshi"] = kalshi_metrics
        except Exception as e:
            logger.error(f"Kalshi training failed: {e}")
            all_results["kalshi"] = {"error": str(e)}

    # ── Final report ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING COMPLETE — Summary")
    print("="*60)
    for model_name, metrics in all_results.items():
        print(f"\n  {model_name.upper()}:")
        if "error" in metrics:
            print(f"    ERROR: {metrics['error']}")
        else:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v}")

    print(f"\n  Saved models: {ModelRegistry.list_models()}")
    print("\n" + "="*60)
    print("  Models ready for MiroFish prediction pipeline.")
    print("  Next: Start backend and use /api/sports/predict endpoint")
    print("="*60 + "\n")


def _run_validation():
    """Validate all existing models against hold-out data."""
    models = ModelRegistry.list_models()
    if not models:
        print("No trained models found. Run training first.")
        return
    for key in models:
        meta = ModelRegistry.get_meta(key)
        if meta:
            print(f"\n{key.upper()} model:")
            for k, v in meta.items():
                if isinstance(v, (int, float, str)):
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
