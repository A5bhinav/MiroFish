"""
MiroFish ML Prediction Layer

Standalone ML models trained on public datasets.
These models feed structured predictions into the MiroFish LLM pipeline,
giving the OASIS simulation hard statistical priors to reason around.

Architecture:
  Public Datasets → Feature Engineering → XGBoost/LightGBM Models
      ↓
  Structured predictions (probabilities, confidence, key factors)
      ↓
  SportsNarrativeFormatter → Zep Graph → OASIS Simulation → Report
      ↓
  ProbabilityExtractor (final LLM calibration)

Sports: NBA, Soccer (EPL, La Liga, Serie A, Bundesliga)
Markets: Moneyline, Spread, Totals, Player Props
Kalshi: Binary prediction market calibration
"""
from .model_registry import ModelRegistry
from .nba_predictor import NBAPredictor
from .soccer_predictor import SoccerPredictor
from .kalshi_predictor import KalshiPredictor

__all__ = ["NBAPredictor", "SoccerPredictor", "KalshiPredictor", "ModelRegistry"]
