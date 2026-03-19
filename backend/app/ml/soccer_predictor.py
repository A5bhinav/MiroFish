"""
Soccer Prediction Model

Two complementary models trained on football-data.co.uk CSVs (2017–2025):
  1. Match outcome classifier (home win / draw / away win)
     — XGBoost with calibration
  2. Total goals model (over/under 2.5 goals)
     — XGBoost regressor

Leagues: EPL, La Liga, Serie A, Bundesliga, Ligue 1
Seasons: 2017/18 through 2024/25 (~15,000 matches)

Key factors (from Dixon-Coles, Shin, and market microstructure research):
  - Market odds (Pinnacle/Bet365): near-unbeatable for top leagues
    BUT we can find edges in: team form divergence, fixture congestion,
    low-profile matches (weaker pricing), and motivational factors.
  - xG form: better than goals for predicting next match
  - Home advantage: varies by league (Bundesliga ~0.35 goals, EPL ~0.40)
  - Rest / fixture congestion: underpriced in markets
  - Head-to-head: significant in derbies and rivalry matches
  - Squad depth: matters when teams play 3 games/week (not in our data but noted)

IMPORTANT on calibration:
  Soccer has 3 outcomes. We train a binary (home win vs. not) model AND
  separately model draw probability as: P(draw) ≈ P(not home win) - P(away win).
  The market draw implied probability is our best draw baseline.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, accuracy_score, mean_absolute_error
)
import joblib

from .feature_store import SoccerFeatureEngineer
from .data_pipeline import get_soccer_all_leagues, get_soccer_spi
from .model_registry import ModelRegistry

logger = logging.getLogger("mirofish.ml.soccer")
warnings.filterwarnings("ignore")


class SoccerPredictor:
    """
    Predicts soccer match outcomes and totals.

    Usage:
        predictor = SoccerPredictor()
        predictor.train()

        pred = predictor.predict(
            home_team="Arsenal",
            away_team="Chelsea",
            league="EPL",
            home_form={"pts_per_game_L5": 2.2, "goals_scored_L5": 1.8, ...},
            away_form={...},
            b365_odds={"home": 1.9, "draw": 3.4, "away": 4.0},
        )
    """

    MODEL_KEY = "soccer"

    XGB_PARAMS = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,  # Soccer sample smaller, prevent overfitting
        "gamma": 0.2,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self):
        self.model_home_win = None     # P(home win)
        self.model_away_win = None     # P(away win) — combined with home → P(draw)
        self.model_over25 = None       # P(over 2.5 goals)
        self.model_goals = None        # Expected total goals (regression)
        self.feature_cols = []
        self.is_trained = False
        self.training_stats = {}
        self.league_home_rates = {}    # Empirical home win rate per league

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Download data and train all soccer models."""
        logger.info("Training soccer models...")

        # Load data
        df = get_soccer_all_leagues()
        if df is None or len(df) < 500:
            raise RuntimeError("Failed to load soccer data.")

        spi_df = get_soccer_spi()  # Optional enrichment
        logger.info(f"Soccer data: {len(df):,} matches")

        # Compute league-level home win rates
        if "league" in df.columns and "FTR" in df.columns:
            self.league_home_rates = df.groupby("league").apply(
                lambda x: (x["FTR"] == "H").mean()
            ).to_dict()

        # Build features
        logger.info("Building features (this takes a few minutes)...")
        X, meta = SoccerFeatureEngineer.build_features(df, spi_df)
        self.feature_cols = list(X.columns)

        y_home_win = meta["_home_win"]
        y_away_win = (meta["_result"] == "A").astype(int)
        y_over25 = meta["_over_2_5"]
        y_goals = meta["_total_goals"].astype(float)

        # Temporal split: last 2000 matches for test
        n = len(X)
        test_size = min(2000, int(n * 0.15))
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        hw_train, hw_test = y_home_win.iloc[:-test_size], y_home_win.iloc[-test_size:]
        aw_train, aw_test = y_away_win.iloc[:-test_size], y_away_win.iloc[-test_size:]
        o25_train, o25_test = y_over25.iloc[:-test_size], y_over25.iloc[-test_size:]
        goals_train, goals_test = y_goals.iloc[:-test_size], y_goals.iloc[-test_size:]

        logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

        # Train home win model
        logger.info("Training home win model...")
        base_hw = xgb.XGBClassifier(**self.XGB_PARAMS, eval_metric="logloss", use_label_encoder=False)
        base_hw.fit(X_train, hw_train, eval_set=[(X_test, hw_test)], verbose=False)
        self.model_home_win = CalibratedClassifierCV(base_hw, method="isotonic", cv=3)
        self.model_home_win.fit(X_train, hw_train)

        # Train away win model
        logger.info("Training away win model...")
        base_aw = xgb.XGBClassifier(**self.XGB_PARAMS, eval_metric="logloss", use_label_encoder=False)
        base_aw.fit(X_train, aw_train, eval_set=[(X_test, aw_test)], verbose=False)
        self.model_away_win = CalibratedClassifierCV(base_aw, method="isotonic", cv=3)
        self.model_away_win.fit(X_train, aw_train)

        # Train over 2.5 model
        logger.info("Training over/under 2.5 model...")
        base_o25 = xgb.XGBClassifier(**self.XGB_PARAMS, eval_metric="logloss", use_label_encoder=False)
        base_o25.fit(X_train, o25_train, eval_set=[(X_test, o25_test)], verbose=False)
        self.model_over25 = CalibratedClassifierCV(base_o25, method="isotonic", cv=3)
        self.model_over25.fit(X_train, o25_train)

        # Train total goals regression
        logger.info("Training goals model...")
        goals_params = {**self.XGB_PARAMS, "objective": "reg:squarederror", "eval_metric": "rmse"}
        goals_params.pop("use_label_encoder", None)
        self.model_goals = xgb.XGBRegressor(**goals_params)
        self.model_goals.fit(X_train, goals_train, eval_set=[(X_test, goals_test)], verbose=False)

        # Evaluate
        metrics = self._evaluate(X_test, hw_test, aw_test, o25_test, goals_test)
        self.training_stats = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": self.feature_cols,
            "league_home_rates": self.league_home_rates,
            **metrics
        }
        self.is_trained = True
        logger.info(f"Soccer training complete. Metrics: {metrics}")
        return metrics

    def _evaluate(self, X_test, hw_test, aw_test, o25_test, goals_test) -> Dict:
        metrics = {}

        hw_probs = self.model_home_win.predict_proba(X_test)[:, 1]
        aw_probs = self.model_away_win.predict_proba(X_test)[:, 1]

        metrics["home_win_auc"] = round(roc_auc_score(hw_test, hw_probs), 4)
        metrics["home_win_brier"] = round(brier_score_loss(hw_test, hw_probs), 4)
        metrics["away_win_auc"] = round(roc_auc_score(aw_test, aw_probs), 4)

        o25_probs = self.model_over25.predict_proba(X_test)[:, 1]
        metrics["over25_auc"] = round(roc_auc_score(o25_test, o25_probs), 4)
        metrics["over25_brier"] = round(brier_score_loss(o25_test, o25_probs), 4)

        goals_pred = self.model_goals.predict(X_test)
        metrics["goals_mae"] = round(mean_absolute_error(goals_test, goals_pred), 3)

        # Calibration: compare our probabilities to Bet365 implied
        # Market is our best baseline for big leagues
        b365_hw = X_test["b365_home_implied_prob"].values
        metrics["b365_home_brier"] = round(brier_score_loss(hw_test, b365_hw), 4)
        metrics["model_vs_market_improvement"] = round(
            metrics["b365_home_brier"] - metrics["home_win_brier"], 4
        )

        return metrics

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def predict(self,
                home_team: str,
                away_team: str,
                league: str = "EPL",
                home_form: Optional[Dict] = None,
                away_form: Optional[Dict] = None,
                h2h: Optional[Dict] = None,
                b365_odds: Optional[Dict] = None,
                spi_home: float = 50.0,
                spi_away: float = 50.0,
                home_rest_days: int = 7,
                away_rest_days: int = 7) -> Dict[str, Any]:
        """
        Generate predictions for an upcoming soccer match.

        Args:
            home_team, away_team: Team names
            league: League name ("EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1")
            home_form: Dict with keys matching SoccerFeatureEngineer output
            away_form: Dict with keys matching SoccerFeatureEngineer output
            h2h: Dict with h2h_home_win_rate, h2h_avg_goals
            b365_odds: {"home": float, "draw": float, "away": float} decimal odds
            spi_home, spi_away: FiveThirtyEight Soccer Power Index ratings
            home_rest_days, away_rest_days: Days since last match

        Returns dict with:
            home_win_prob, draw_prob, away_win_prob
            over_2_5_prob, expected_goals
            confidence, key_factors
        """
        if not self.is_trained:
            if not self.load():
                return self._fallback_predict(home_team, away_team, b365_odds, league)

        # Default form if not provided
        home_form = home_form or {}
        away_form = away_form or {}
        h2h = h2h or {"home_win_rate": 0.4, "avg_goals": 2.5}
        b365_odds = b365_odds or {"home": 2.0, "draw": 3.4, "away": 3.5}

        # Normalize odds to probabilities
        ph, pd_, pa = SoccerFeatureEngineer._odds_to_prob(
            b365_odds.get("home", 2.0),
            b365_odds.get("draw", 3.4),
            b365_odds.get("away", 3.5)
        )

        features = {
            "home_pts_per_game_L5": home_form.get("pts_per_game_L5", 1.5),
            "home_pts_per_game_L10": home_form.get("pts_per_game_L10", 1.5),
            "home_goals_scored_avg_L5": home_form.get("goals_scored_L5", 1.4),
            "home_goals_conceded_avg_L5": home_form.get("goals_conceded_L5", 1.2),
            "home_goal_diff_L5": home_form.get("goal_diff_L5", 0.2),
            "home_shots_on_target_avg_L5": home_form.get("shots_on_target_L5", 5.0),
            "home_win_streak": home_form.get("win_streak", 0),
            "away_pts_per_game_L5": away_form.get("pts_per_game_L5", 1.5),
            "away_pts_per_game_L10": away_form.get("pts_per_game_L10", 1.5),
            "away_goals_scored_avg_L5": away_form.get("goals_scored_L5", 1.4),
            "away_goals_conceded_avg_L5": away_form.get("goals_conceded_L5", 1.2),
            "away_goal_diff_L5": away_form.get("goal_diff_L5", 0.2),
            "away_shots_on_target_avg_L5": away_form.get("shots_on_target_L5", 5.0),
            "away_win_streak": away_form.get("win_streak", 0),
            "pts_per_game_diff": home_form.get("pts_per_game_L5", 1.5) - away_form.get("pts_per_game_L5", 1.5),
            "goals_scored_diff": home_form.get("goals_scored_L5", 1.4) - away_form.get("goals_scored_L5", 1.4),
            "goals_conceded_diff": home_form.get("goals_conceded_L5", 1.2) - away_form.get("goals_conceded_L5", 1.2),
            "h2h_home_win_rate": h2h.get("home_win_rate", 0.4),
            "h2h_avg_goals": h2h.get("avg_goals", 2.5),
            "b365_home_implied_prob": ph,
            "b365_draw_implied_prob": pd_,
            "b365_away_implied_prob": pa,
            "market_home_minus_draw_prob": ph - pd_,
            "is_top6": 0,
            "days_since_last_match_home": min(home_rest_days, 14),
            "days_since_last_match_away": min(away_rest_days, 14),
            "rest_advantage": min(home_rest_days, 14) - min(away_rest_days, 14),
            "spi_diff": spi_home - spi_away,
            "spi_prob_home": ph,  # Use market as SPI proxy if not available
        }

        X = pd.DataFrame([features])
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_cols].fillna(0)

        home_win_prob = float(self.model_home_win.predict_proba(X)[0, 1])
        away_win_prob = float(self.model_away_win.predict_proba(X)[0, 1])
        # Draw: residual after normalizing home + away
        raw_draw = max(0, 1.0 - home_win_prob - away_win_prob)
        total = home_win_prob + away_win_prob + raw_draw
        home_win_prob /= total
        away_win_prob /= total
        draw_prob = raw_draw / total

        over25_prob = float(self.model_over25.predict_proba(X)[0, 1])
        expected_goals = float(self.model_goals.predict(X)[0])

        # Confidence
        market_edge = abs(home_win_prob - ph)
        if market_edge > 0.05:
            confidence = "medium"
        elif market_edge > 0.10:
            confidence = "high"
        else:
            confidence = "low"  # Model agrees with market — no edge identified

        # Key factors
        key_factors = []
        market_fav = "Home" if ph > pa else "Away"
        model_fav = "Home" if home_win_prob > away_win_prob else "Away"

        if market_fav != model_fav:
            key_factors.append(
                f"Model DISAGREES with market: market favors {market_fav} ({ph:.0%}), "
                f"model favors {model_fav} ({max(home_win_prob, away_win_prob):.0%}) — potential edge"
            )
        else:
            key_factors.append(
                f"Market and model agree: {model_fav} team win probability {max(home_win_prob, away_win_prob):.0%}"
            )

        home_pts = home_form.get("pts_per_game_L5", 1.5)
        away_pts = away_form.get("pts_per_game_L5", 1.5)
        form_diff = home_pts - away_pts
        if abs(form_diff) > 0.5:
            better = home_team if form_diff > 0 else away_team
            key_factors.append(f"{better} in significantly better form (L5 pts/game diff: {abs(form_diff):.1f})")

        if h2h.get("home_win_rate", 0.4) > 0.6:
            key_factors.append(f"{home_team} dominates H2H record ({h2h['home_win_rate']:.0%} win rate)")
        elif h2h.get("home_win_rate", 0.4) < 0.25:
            key_factors.append(f"{away_team} has strong H2H advantage ({1-h2h['home_win_rate']:.0%} win rate)")

        if expected_goals > 3.0:
            key_factors.append(f"High-scoring matchup expected: {expected_goals:.1f} total goals")
        elif expected_goals < 2.0:
            key_factors.append(f"Low-scoring matchup expected: {expected_goals:.1f} total goals")

        league_rate = self.league_home_rates.get(league, 0.45)
        key_factors.append(f"{league} home advantage: {league_rate:.0%} historic home win rate")

        return {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "home_win_prob": round(home_win_prob, 4),
            "draw_prob": round(draw_prob, 4),
            "away_win_prob": round(away_win_prob, 4),
            "over_2_5_prob": round(over25_prob, 4),
            "expected_goals": round(expected_goals, 2),
            "market_home_implied_prob": round(ph, 4),
            "market_away_implied_prob": round(pa, 4),
            "confidence": confidence,
            "key_factors": key_factors,
        }

    def _fallback_predict(self, home_team, away_team, b365_odds, league):
        """Market-only fallback."""
        b365_odds = b365_odds or {"home": 2.0, "draw": 3.4, "away": 3.5}
        ph, pd_, pa = SoccerFeatureEngineer._odds_to_prob(
            b365_odds.get("home", 2.0),
            b365_odds.get("draw", 3.4),
            b365_odds.get("away", 3.5)
        )
        return {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "home_win_prob": round(ph, 4),
            "draw_prob": round(pd_, 4),
            "away_win_prob": round(pa, 4),
            "over_2_5_prob": 0.52,
            "expected_goals": 2.5,
            "confidence": "low",
            "key_factors": ["Market odds only (model not trained)"],
        }

    def save(self):
        ModelRegistry.save(self.MODEL_KEY, {
            "model_home_win": self.model_home_win,
            "model_away_win": self.model_away_win,
            "model_over25": self.model_over25,
            "model_goals": self.model_goals,
            "feature_cols": self.feature_cols,
            "league_home_rates": self.league_home_rates,
            "training_stats": self.training_stats,
        })

    def load(self) -> bool:
        data = ModelRegistry.load(self.MODEL_KEY)
        if data is None:
            return False
        self.model_home_win = data["model_home_win"]
        self.model_away_win = data["model_away_win"]
        self.model_over25 = data["model_over25"]
        self.model_goals = data["model_goals"]
        self.feature_cols = data["feature_cols"]
        self.league_home_rates = data.get("league_home_rates", {})
        self.training_stats = data.get("training_stats", {})
        self.is_trained = True
        return True
