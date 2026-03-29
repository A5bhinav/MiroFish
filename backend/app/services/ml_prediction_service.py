"""
ML Prediction Service

Bridges the ML prediction models (NBA, Soccer, Kalshi) with the MiroFish
LLM pipeline. This service:

  1. Loads trained models (lazy, first-use initialization)
  2. Runs ML predictions given a SportConfig or Kalshi market question
  3. Formats predictions as rich narrative text for Zep ingestion
     (so the OASIS simulation agents have hard statistical priors to reason around)
  4. Returns structured probability dicts for direct API responses

Integration points:
  - Called from sports.py ingest task (enriches narrative with ML predictions)
  - Called from report.py (passes ML context to ReportAgent)
  - Called from probability_extractor.py (provides initial probability anchor)
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger("mirofish.ml_service")

# Lazy imports — models only loaded when first used
_nba_predictor = None
_soccer_predictor = None
_kalshi_predictor = None


def _get_nba() :
    global _nba_predictor
    if _nba_predictor is None:
        from ..ml.nba_predictor import NBAPredictor
        _nba_predictor = NBAPredictor()
        _nba_predictor.load()
    return _nba_predictor


def _get_soccer():
    global _soccer_predictor
    if _soccer_predictor is None:
        from ..ml.soccer_predictor import SoccerPredictor
        _soccer_predictor = SoccerPredictor()
        _soccer_predictor.load()
    return _soccer_predictor


def _get_kalshi():
    global _kalshi_predictor
    if _kalshi_predictor is None:
        from ..ml.kalshi_predictor import KalshiPredictor
        _kalshi_predictor = KalshiPredictor()
        if not _kalshi_predictor.load():
            _kalshi_predictor._build_fallback_calibrator()
    return _kalshi_predictor


# ---------------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------------

def predict_nba_game(sport_config: Dict[str, Any],
                      raw_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Generate ML predictions for an NBA game given a SportConfig dict.

    Args:
        sport_config: The project's sport_config dict
        raw_data: Optional raw data from SportsDataOrchestrator (for rolling features)

    Returns structured prediction dict.
    """
    predictor = _get_nba()

    team_a = sport_config.get("team_a_name", "Home Team")
    team_b = sport_config.get("team_b_name", "Away Team")

    # Try to extract ELO from raw_data if available
    home_elo = 1500.0
    away_elo = 1500.0
    home_rest = 2
    away_rest = 2

    if raw_data:
        # Extract rest days from most recent games
        games_a = raw_data.get("recent_games_a", [])
        games_b = raw_data.get("recent_games_b", [])
        if games_a:
            try:
                import pandas as pd
                # BDL sorts games ascending by date — use max() to get most recent
                most_recent_a = max(games_a, key=lambda g: g.get("date", ""))
                last_a = pd.to_datetime(most_recent_a.get("date", ""))
                today = pd.Timestamp.now()
                home_rest = max(0, min(7, (today - last_a).days))
            except Exception:
                pass
        if games_b:
            try:
                import pandas as pd
                most_recent_b = max(games_b, key=lambda g: g.get("date", ""))
                last_b = pd.to_datetime(most_recent_b.get("date", ""))
                today = pd.Timestamp.now()
                away_rest = max(0, min(7, (today - last_b).days))
            except Exception:
                pass

    prediction = predictor.predict(
        home_team=team_a,
        away_team=team_b,
        home_elo=home_elo,
        away_elo=away_elo,
        home_rest_days=home_rest,
        away_rest_days=away_rest,
        is_playoffs=False,
        season=2025
    )

    return prediction


def predict_soccer_game(sport_config: Dict[str, Any],
                         raw_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Generate ML predictions for a soccer match.
    """
    predictor = _get_soccer()

    team_a = sport_config.get("team_a_name", "Home Team")
    team_b = sport_config.get("team_b_name", "Away Team")
    league = sport_config.get("league", "EPL")

    prediction = predictor.predict(
        home_team=team_a,
        away_team=team_b,
        league=league,
    )
    return prediction


def predict_kalshi(market_question: str,
                    current_market_price: float = 0.5,
                    time_to_close_days: float = 30.0,
                    category: str = "economics") -> Dict[str, Any]:
    """
    Generate calibrated probability for a Kalshi market question.

    Args:
        market_question: Full text of the YES/NO question
        current_market_price: Current Kalshi market price (0–1) — use 0.5 if unknown
        time_to_close_days: Days until market resolves
        category: "economics", "politics", "sports", "weather", "other"
    """
    predictor = _get_kalshi()
    econ_context = predictor.get_economic_context()
    prediction = predictor.predict(
        market_question=market_question,
        community_prob=current_market_price,
        category=category,
        time_to_close_days=time_to_close_days,
        economic_context=econ_context,
    )
    return prediction


# ---------------------------------------------------------------------------
# Narrative formatting (fed into Zep graph)
# ---------------------------------------------------------------------------

def format_nba_ml_narrative(prediction: Dict[str, Any], sport_config: Dict) -> str:
    """
    Convert NBA ML prediction to rich narrative text for Zep ingestion.
    This text is added to the knowledge graph so simulation agents
    can reason with hard statistical priors.
    """
    home = prediction.get("home_team", sport_config.get("team_a_name", "Home"))
    away = prediction.get("away_team", sport_config.get("team_b_name", "Away"))
    ml_prob = prediction.get("moneyline_prob", 0.5)
    spread = prediction.get("spread_prediction", 0)
    total = prediction.get("total_prediction", 220)
    conf = prediction.get("confidence", "low")
    factors = prediction.get("key_factors", [])
    warning = prediction.get("warning", "")

    lines = [
        f"ML STATISTICAL PREDICTION — {home} vs {away}",
        "",
        f"MONEYLINE: {home} win probability {ml_prob:.1%} (model confidence: {conf.upper()})",
        f"  Implied odds for {home}: {_prob_to_american(ml_prob):+d}",
        f"  Implied odds for {away}: {_prob_to_american(1-ml_prob):+d}",
        "",
        f"SPREAD: {home} projected to {'win' if spread > 0 else 'lose'} by "
        f"{abs(spread):.1f} points",
        f"  This implies a spread of {home} {-spread:+.1f}",
        "",
        f"TOTAL: Projected combined score {total:.0f} points",
        f"  Over/under line context: league average ~218 pts",
        "",
        "STATISTICAL FACTORS:",
    ]
    for f in factors:
        lines.append(f"  • {f}")

    if warning:
        lines.append(f"\n⚠️  WARNING: {warning}")

    elo_diff = prediction.get("elo_diff", 0)
    lines.extend([
        "",
        f"ELO RATINGS: {home} {prediction.get('home_elo', 1500):.0f} | "
        f"{away} {prediction.get('away_elo', 1500):.0f} (diff: {elo_diff:+.0f})",
        "",
        "NOTE: These are pre-game statistical estimates based on historical ELO ratings",
        "and situational factors. They do not account for lineup changes, last-minute",
        "injury reports, or motivational factors — which the simulation agents will assess.",
    ])

    return "\n".join(lines)


def format_soccer_ml_narrative(prediction: Dict[str, Any], sport_config: Dict) -> str:
    """Format soccer ML predictions as narrative text."""
    home = prediction.get("home_team", sport_config.get("team_a_name", "Home"))
    away = prediction.get("away_team", sport_config.get("team_b_name", "Away"))
    hw = prediction.get("home_win_prob", 0.4)
    dw = prediction.get("draw_prob", 0.3)
    aw = prediction.get("away_win_prob", 0.3)
    o25 = prediction.get("over_2_5_prob", 0.5)
    xg = prediction.get("expected_goals", 2.5)
    conf = prediction.get("confidence", "low")
    factors = prediction.get("key_factors", [])
    market_hw = prediction.get("market_home_implied_prob", hw)
    market_aw = prediction.get("market_away_implied_prob", aw)

    lines = [
        f"ML STATISTICAL PREDICTION — {home} vs {away}",
        "",
        "MATCH OUTCOME PROBABILITIES (1X2):",
        f"  Home Win ({home}):  {hw:.1%}  [market: {market_hw:.1%}]",
        f"  Draw:                {dw:.1%}",
        f"  Away Win ({away}): {aw:.1%}  [market: {market_aw:.1%}]",
        "",
        "GOALS MARKETS:",
        f"  Expected total goals: {xg:.2f}",
        f"  Over 2.5 goals probability: {o25:.1%}",
        f"  Under 2.5 goals probability: {1-o25:.1%}",
        "",
        f"MODEL CONFIDENCE: {conf.upper()}",
        "",
        "STATISTICAL FACTORS:",
    ]
    for f in factors:
        lines.append(f"  • {f}")

    # Market edge analysis
    hw_edge = hw - market_hw
    aw_edge = aw - market_aw
    if abs(hw_edge) > 0.04 or abs(aw_edge) > 0.04:
        lines.extend([
            "",
            "MARKET EDGE ANALYSIS:",
            f"  Model vs market for {home}: {hw_edge:+.1%}",
            f"  Model vs market for {away}: {aw_edge:+.1%}",
            "  (Positive = model higher than market; potential value bet)",
        ])

    lines.extend([
        "",
        "NOTE: Market odds from Bet365/Pinnacle are already highly efficient for top leagues.",
        "The largest edge opportunities come from: fixture congestion, low-profile matches,",
        "and team-specific factors the market underweights (e.g., squad rotation).",
    ])
    return "\n".join(lines)


def format_kalshi_ml_narrative(prediction: Dict[str, Any]) -> str:
    """Format Kalshi ML predictions as narrative text."""
    yes_prob = prediction.get("yes_probability", 0.5)
    no_prob = prediction.get("no_probability", 0.5)
    raw = prediction.get("raw_market_prob", 0.5)
    adj = prediction.get("calibration_adjustment", 0)
    edge = prediction.get("edge_signal", "NEUTRAL")
    kelly = prediction.get("kelly_fraction", 0)
    bet_size = prediction.get("suggested_bet_size", "skip")
    factors = prediction.get("factors", [])
    base_rate = prediction.get("base_rate")

    lines = [
        "ML CALIBRATION ANALYSIS — KALSHI PREDICTION MARKET",
        "",
        f"RAW MARKET PRICE: {raw:.1%} YES",
        f"CALIBRATED PROBABILITY: {yes_prob:.1%} YES / {no_prob:.1%} NO",
        f"CALIBRATION ADJUSTMENT: {adj:+.1%}",
        "",
        f"EDGE SIGNAL: {edge}",
        f"KELLY FRACTION: {kelly:.1%} of bankroll (quarter-Kelly sizing)",
        f"SUGGESTED POSITION SIZE: {bet_size.upper()}",
        "",
    ]

    if base_rate:
        lines.extend([
            f"HISTORICAL BASE RATE: {base_rate:.1%} for this question type",
            "",
        ])

    lines.append("ANALYSIS FACTORS:")
    for f in factors:
        lines.append(f"  • {f}")

    lines.extend([
        "",
        "IMPORTANT: Kalshi markets are regulated prediction markets. Our calibration",
        "model is based on historical Metaculus predictions and economic indicators.",
        "The biggest edges come from: (1) low-liquidity markets with weak pricing,",
        "(2) questions where you have genuine information the market hasn't priced in,",
        "(3) markets anchoring too heavily on recent events vs. long-run base rates.",
        "",
        "⚠️  NEVER bet more than you can afford to lose. Markets can stay 'wrong' longer",
        "than your bankroll can stay solvent.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _prob_to_american(prob: float) -> int:
    """Convert win probability to American odds (moneyline)."""
    prob = max(0.01, min(0.99, prob))
    if prob >= 0.5:
        return -int(round(prob / (1 - prob) * 100))
    else:
        return int(round((1 - prob) / prob * 100))


def enrich_sports_narrative_with_ml(document_texts: List[str],
                                      sport_config: Dict[str, Any],
                                      raw_data: Optional[Dict] = None) -> List[str]:
    """
    Appends ML prediction narrative to the existing sports document texts.
    Called from the sports ingest pipeline before Zep graph build.
    """
    try:
        sport = sport_config.get("sport", "nba").lower()
        if sport == "nba":
            pred = predict_nba_game(sport_config, raw_data)
            narrative = format_nba_ml_narrative(pred, sport_config)
        elif sport in ("soccer", "football"):
            pred = predict_soccer_game(sport_config, raw_data)
            narrative = format_soccer_ml_narrative(pred, sport_config)
        else:
            return document_texts

        logger.info(f"Appended ML prediction narrative ({len(narrative)} chars)")
        return document_texts + [narrative]
    except Exception as e:
        logger.warning(f"ML enrichment failed (non-fatal): {e}")
        return document_texts
