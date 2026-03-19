"""
NBA Prediction Model

Trains three XGBoost models on historical game data:
  1. Moneyline (win/loss) — home team win probability
  2. Point spread — expected point differential (home - away)
  3. Total (over/under) — expected total points

Primary training data: FiveThirtyEight NBA ELO (2000–2025, ~20k games)
Secondary: nba_api game logs (last 10 seasons, for rolling features)

Key factors incorporated:
  - ELO ratings (best proven long-run predictor)
  - RAPTOR/CARMELO ratings when available
  - Rolling form: win%, net rating, scoring trend
  - Situational: home court, back-to-back, rest days
  - Playoff vs regular season (different dynamics)
  - Pace-adjusted stats for total predictions

Validation target:
  - Moneyline AUC > 0.72
  - Brier score < 0.22
  - Beating naive baseline (always predict ELO probability) on test set
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, accuracy_score, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
import joblib

from .feature_store import NBAFeatureEngineer
from .data_pipeline import get_nba_elo, get_nba_all_seasons_game_logs
from .model_registry import ModelRegistry

logger = logging.getLogger("mirofish.ml.nba")
warnings.filterwarnings("ignore")


class NBAPredictor:
    """
    Predicts NBA game outcomes across moneyline, spread, and totals.

    Usage:
        predictor = NBAPredictor()
        predictor.train()           # downloads data + trains models
        predictor.save()            # saves to app/ml/models/

        # Inference:
        pred = predictor.predict(
            home_team="Boston Celtics",
            away_team="Miami Heat",
            home_elo=1650, away_elo=1520,
            home_rest_days=2, away_rest_days=1,
            is_playoffs=False,
        )
        # Returns: {moneyline: 0.68, spread: -4.5, total: 221.3, confidence: "high", ...}
    """

    MODEL_KEY = "nba"

    # XGBoost hyperparameters tuned for sports prediction
    XGB_PARAMS = {
        "n_estimators": 500,
        "max_depth": 4,            # Shallow trees prevent overfitting on small dataset
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,     # Avoid fitting outlier games
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "n_jobs": -1,
    }

    def __init__(self):
        self.model_ml = None        # Moneyline classifier
        self.model_spread = None    # Point differential regressor
        self.model_total = None     # Total points regressor
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.is_trained = False
        self.training_stats = {}

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(self, use_api: bool = False) -> Dict[str, Any]:
        """
        Train all three models.
        Returns dict of validation metrics.

        Args:
            use_api: If True, also pull nba_api game logs (slower but richer features).
                     If False, use only FiveThirtyEight ELO (fast, good accuracy).
        """
        logger.info("Training NBA models...")

        # 1. Load data
        elo_df = get_nba_elo()
        if elo_df is None or len(elo_df) < 100:
            raise RuntimeError("Failed to load NBA ELO data. Check internet connection.")

        logger.info(f"Loaded {len(elo_df):,} NBA games for training")

        # 2. Build feature matrix
        df, X = NBAFeatureEngineer.build_features_from_elo(elo_df)

        # Target variables
        y_ml = df["home_win"]
        y_spread = df["point_diff"]
        y_total = df["total_pts"]

        self.feature_cols = list(X.columns)
        logger.info(f"Features: {self.feature_cols}")

        # 3. Time-series split (never train on future to predict past)
        # Use last 2 seasons as test set
        n = len(df)
        test_size = min(2000, int(n * 0.15))  # ~15% for test
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_ml_train, y_ml_test = y_ml.iloc[:-test_size], y_ml.iloc[-test_size:]
        y_spread_train, y_spread_test = y_spread.iloc[:-test_size], y_spread.iloc[-test_size:]
        y_total_train, y_total_test = y_total.iloc[:-test_size], y_total.iloc[-test_size:]

        logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

        # 4. Train moneyline model
        logger.info("Training moneyline model...")
        base_ml = xgb.XGBClassifier(**self.XGB_PARAMS)
        base_ml.fit(
            X_train, y_ml_train,
            eval_set=[(X_test, y_ml_test)],
            verbose=False
        )
        # Isotonic calibration — critical for getting honest probabilities
        self.model_ml = CalibratedClassifierCV(base_ml, method="isotonic", cv=3)
        self.model_ml.fit(X_train, y_ml_train)

        # 5. Train spread model (regression)
        logger.info("Training spread model...")
        spread_params = {**self.XGB_PARAMS,
                          "objective": "reg:squarederror",
                          "eval_metric": "rmse"}
        spread_params.pop("use_label_encoder", None)
        self.model_spread = xgb.XGBRegressor(**spread_params)
        self.model_spread.fit(
            X_train, y_spread_train,
            eval_set=[(X_test, y_spread_test)],
            verbose=False
        )

        # 6. Train total model (regression)
        logger.info("Training total points model...")
        self.model_total = xgb.XGBRegressor(**spread_params)
        self.model_total.fit(
            X_train, y_total_train,
            eval_set=[(X_test, y_total_test)],
            verbose=False
        )

        # 7. Evaluate
        metrics = self._evaluate(X_test, y_ml_test, y_spread_test, y_total_test)
        self.training_stats = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": self.feature_cols,
            **metrics
        }
        self.is_trained = True
        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics

    def _evaluate(self, X_test, y_ml, y_spread, y_total) -> Dict:
        """Compute test-set metrics for all three models."""
        metrics = {}

        # Moneyline
        ml_probs = self.model_ml.predict_proba(X_test)[:, 1]
        metrics["ml_auc"] = round(roc_auc_score(y_ml, ml_probs), 4)
        metrics["ml_brier"] = round(brier_score_loss(y_ml, ml_probs), 4)
        metrics["ml_logloss"] = round(log_loss(y_ml, ml_probs), 4)
        metrics["ml_accuracy"] = round(accuracy_score(y_ml, (ml_probs > 0.5).astype(int)), 4)

        # Baseline comparison (always predict 0.5)
        metrics["ml_brier_baseline"] = round(brier_score_loss(y_ml, np.full(len(y_ml), 0.5)), 4)
        metrics["ml_brier_improvement"] = round(metrics["ml_brier_baseline"] - metrics["ml_brier"], 4)

        # Spread
        spread_pred = self.model_spread.predict(X_test)
        metrics["spread_mae"] = round(mean_absolute_error(y_spread, spread_pred), 2)
        metrics["spread_rmse"] = round(float(np.sqrt(np.mean((y_spread - spread_pred)**2))), 2)

        # Total
        total_pred = self.model_total.predict(X_test)
        metrics["total_mae"] = round(mean_absolute_error(y_total, total_pred), 2)
        metrics["total_rmse"] = round(float(np.sqrt(np.mean((y_total - total_pred)**2))), 2)

        # Simulated betting ROI at >60% confidence threshold
        mask = (ml_probs > 0.60) | (ml_probs < 0.40)
        if mask.sum() > 50:
            high_conf = ml_probs[mask]
            high_y = np.array(y_ml)[mask]
            pred_side = (high_conf > 0.5).astype(int)
            win_rate = accuracy_score(high_y, pred_side)
            # Assuming -110 odds (American standard): need >52.4% to profit
            roi = (win_rate * 1.909 - 1) * 100  # ROI% at -110 odds
            metrics["high_conf_games"] = int(mask.sum())
            metrics["high_conf_win_rate"] = round(win_rate, 4)
            metrics["high_conf_roi_pct"] = round(roi, 2)

        return metrics

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def predict(self,
                home_team: str = "",
                away_team: str = "",
                home_elo: float = 1500.0,
                away_elo: float = 1500.0,
                home_rest_days: int = 2,
                away_rest_days: int = 2,
                is_playoffs: bool = False,
                is_neutral: bool = False,
                season: int = 2025,
                raptor_diff: float = 0.0,
                carmelo_diff: float = 0.0) -> Dict[str, Any]:
        """
        Generate predictions for an upcoming game.

        Args:
            home_team, away_team: Team names (for labeling)
            home_elo, away_elo: Current ELO ratings (get from nba_api or our cache)
            home_rest_days, away_rest_days: Days since last game (0=B2B, 7+=fresh)
            is_playoffs: Changes dynamics significantly
            raptor_diff, carmelo_diff: Advanced metric differentials if available

        Returns dict with:
            moneyline_prob: float — home team win probability (0–1)
            spread_prediction: float — expected home - away point differential
            total_prediction: float — expected total points
            confidence: str — "high" / "medium" / "low"
            key_factors: List[str] — top factors driving the prediction
            warning: str — any flags (B2B, extreme ELO mismatch, etc.)
        """
        if not self.is_trained:
            if not self.load():
                return self._fallback_predict(home_elo, away_elo, home_team, away_team)

        elo_diff = home_elo - away_elo
        elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))

        # Home court adjustment: ~3.5 pt / ~0.06 probability
        home_court_boost = 0.0 if is_neutral else 0.06
        adjusted_elo_prob = min(0.95, max(0.05, elo_prob + home_court_boost))

        # Rest adjustment
        rest_diff = home_rest_days - away_rest_days
        home_b2b = 1 if home_rest_days <= 1 else 0
        away_b2b = 1 if away_rest_days <= 1 else 0

        # Look up current team rolling stats from cached dataset
        home_win_pct_L10 = 0.5
        away_win_pct_L10 = 0.5
        home_win_pct_L5 = 0.5
        away_win_pct_L5 = 0.5
        home_pts_avg_L10 = 110.0
        away_pts_avg_L10 = 110.0
        home_pm_avg_L10 = 0.0
        away_pm_avg_L10 = 0.0
        home_fg_pct_L10 = 0.46
        away_fg_pct_L10 = 0.46

        try:
            from .data_pipeline import get_nba_elo
            cache_df = get_nba_elo()
            if cache_df is not None and len(cache_df) > 0:
                # Get the most recent stats for each team
                cache_df = cache_df.sort_values("date")
                home_rows = cache_df[cache_df["team1"].str.lower() == home_team.lower()]
                away_rows = cache_df[cache_df["team2"].str.lower() == away_team.lower()]
                # Also check as team2 for away team
                if len(home_rows) == 0:
                    home_rows = cache_df[cache_df["team2"].str.lower() == home_team.lower()]
                if len(away_rows) == 0:
                    away_rows = cache_df[cache_df["team1"].str.lower() == away_team.lower()]

                if len(home_rows) > 0:
                    last_h = home_rows.iloc[-1]
                    home_elo = last_h.get("elo1_pre", home_elo)
                    home_win_pct_L10 = last_h.get("home_win_pct_L10", 0.5)
                    home_win_pct_L5 = last_h.get("home_win_pct_L5", 0.5)
                    home_pts_avg_L10 = last_h.get("home_pts_avg_L10", 110.0)
                    home_pm_avg_L10 = last_h.get("home_pm_avg_L10", 0.0)
                    home_fg_pct_L10 = last_h.get("home_fg_pct_L10", 0.46)

                if len(away_rows) > 0:
                    last_a = away_rows.iloc[-1]
                    away_elo = last_a.get("elo2_pre", away_elo)
                    away_win_pct_L10 = last_a.get("away_win_pct_L10", 0.5)
                    away_win_pct_L5 = last_a.get("away_win_pct_L5", 0.5)
                    away_pts_avg_L10 = last_a.get("away_pts_avg_L10", 110.0)
                    away_pm_avg_L10 = last_a.get("away_pm_avg_L10", 0.0)
                    away_fg_pct_L10 = last_a.get("away_fg_pct_L10", 0.46)

                # Recompute ELO diff and prob after lookup
                elo_diff = home_elo - away_elo
                elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))
                adjusted_elo_prob = min(0.95, max(0.05, elo_prob + home_court_boost))
        except Exception:
            pass  # Fallback to defaults on any error

        X = pd.DataFrame([{
            "elo_diff": elo_diff,
            "elo_prob_home": adjusted_elo_prob,
            "raptor_diff": raptor_diff,
            "carmelo_diff": carmelo_diff,
            "is_neutral": int(is_neutral),
            "is_playoffs": int(is_playoffs),
            "season_frac": (season - 2000) / 25.0,
            "home_win_pct_L10": home_win_pct_L10,
            "home_win_pct_L5": home_win_pct_L5,
            "home_pts_avg_L10": home_pts_avg_L10,
            "home_pm_avg_L10": home_pm_avg_L10,
            "home_fg_pct_L10": home_fg_pct_L10,
            "away_win_pct_L10": away_win_pct_L10,
            "away_win_pct_L5": away_win_pct_L5,
            "away_pts_avg_L10": away_pts_avg_L10,
            "away_pm_avg_L10": away_pm_avg_L10,
            "away_fg_pct_L10": away_fg_pct_L10,
            "win_pct_diff_L10": home_win_pct_L10 - away_win_pct_L10,
            "pm_diff_L10": home_pm_avg_L10 - away_pm_avg_L10,
        }])

        # Ensure column order matches training
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_cols].fillna(0)

        ml_prob = float(self.model_ml.predict_proba(X)[0, 1])
        spread = float(self.model_spread.predict(X)[0])
        total = float(self.model_total.predict(X)[0])

        # Confidence based on ELO differential and rest context
        elo_abs = abs(elo_diff)
        if elo_abs > 150 and (home_b2b == away_b2b):
            confidence = "high"
        elif elo_abs > 80:
            confidence = "medium"
        else:
            confidence = "low"

        # Adjust confidence down for B2B games (high variance)
        if home_b2b or away_b2b:
            confidence = "low" if confidence == "high" else "low"

        # Key factors
        key_factors = []
        if elo_abs > 100:
            fav = home_team if elo_diff > 0 else away_team
            key_factors.append(f"Large ELO gap: {fav} favored by {elo_abs:.0f} ELO pts ({ml_prob:.1%} win prob)")
        if home_b2b:
            key_factors.append(f"{home_team} on back-to-back (fatigue penalty ~3 pts)")
        if away_b2b:
            key_factors.append(f"{away_team} on back-to-back (fatigue penalty ~3 pts)")
        if rest_diff > 2:
            key_factors.append(f"{home_team} has {rest_diff} more rest days than {away_team}")
        if is_playoffs:
            key_factors.append("Playoff game: home court advantage amplified, defense elevated")
        if not is_neutral:
            key_factors.append(f"Home court for {home_team} worth ~3.5 pts / +6% win probability")

        warning = ""
        if home_b2b and away_b2b:
            warning = "Both teams on B2B — high variance game, lower confidence"
        elif home_b2b:
            warning = f"{home_team} on B2B — model may underestimate fatigue impact"

        return {
            "home_team": home_team,
            "away_team": away_team,
            "moneyline_prob": round(ml_prob, 4),
            "spread_prediction": round(spread, 1),
            "total_prediction": round(total, 1),
            "confidence": confidence,
            "key_factors": key_factors,
            "elo_diff": round(elo_diff, 0),
            "home_elo": round(home_elo, 0),
            "away_elo": round(away_elo, 0),
            "warning": warning,
        }

    def _fallback_predict(self, home_elo, away_elo, home_team, away_team):
        """Pure ELO-based fallback when model isn't loaded."""
        elo_diff = home_elo - away_elo
        prob = 1 / (1 + 10 ** (-elo_diff / 400)) + 0.04  # +4% home court
        return {
            "home_team": home_team,
            "away_team": away_team,
            "moneyline_prob": round(min(0.95, max(0.05, prob)), 4),
            "spread_prediction": round(elo_diff / 28.0, 1),  # ELO diff → pts
            "total_prediction": 218.0,  # League average
            "confidence": "low",
            "key_factors": ["ELO-only fallback (model not trained)"],
            "warning": "Using ELO fallback — train models for better accuracy",
        }

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self):
        """Save all three models to disk."""
        ModelRegistry.save(self.MODEL_KEY, {
            "model_ml": self.model_ml,
            "model_spread": self.model_spread,
            "model_total": self.model_total,
            "feature_cols": self.feature_cols,
            "training_stats": self.training_stats,
        })
        logger.info(f"NBA models saved to {ModelRegistry.model_path(self.MODEL_KEY)}")

    def load(self) -> bool:
        """Load saved models. Returns True if successful."""
        data = ModelRegistry.load(self.MODEL_KEY)
        if data is None:
            return False
        self.model_ml = data["model_ml"]
        self.model_spread = data["model_spread"]
        self.model_total = data["model_total"]
        self.feature_cols = data["feature_cols"]
        self.training_stats = data.get("training_stats", {})
        self.is_trained = True
        logger.info(f"NBA models loaded. Stats: {self.training_stats}")
        return True

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance for the moneyline model."""
        if not self.is_trained or self.model_ml is None:
            return pd.DataFrame()
        try:
            # Get importance from the base estimator
            base = self.model_ml.calibrated_classifiers_[0].estimator
            importance = base.feature_importances_
            return pd.DataFrame({
                "feature": self.feature_cols,
                "importance": importance
            }).sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()
