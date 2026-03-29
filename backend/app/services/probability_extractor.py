"""
Probability Extractor

After the ReportAgent produces full_report.md, this module reads that
Markdown and runs a single deterministic LLM call (temperature=0.1) to
extract structured probability JSON.

Two flavours:
  - extract_kalshi(report_markdown, market_question) -> dict
  - extract_sports(report_markdown, sport_config)    -> dict
"""

import json
from typing import Any, Dict, List, Optional

from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger("mirofish.probability_extractor")


# ---------------------------------------------------------------------------
# Kalshi / prediction-market probability
# ---------------------------------------------------------------------------

KALSHI_SYSTEM_PROMPT = """You are a calibrated probability estimator.
You will read an AI-generated research report and extract a structured
probability estimate for a binary prediction market question.

Return ONLY a valid JSON object — no markdown, no explanations outside the JSON.

Required JSON shape:
{
  "yes_probability": <float 0.0-1.0>,
  "no_probability": <float 0.0-1.0>,
  "confidence": "<low|medium|high>",
  "key_factors": ["<factor 1>", "<factor 2>", ...],
  "reasoning_summary": "<1-3 sentence justification>"
}

Rules:
- yes_probability + no_probability must equal exactly 1.0
- confidence = "high" if the evidence is strong and consistent
- confidence = "medium" if there are meaningful uncertainties
- confidence = "low" if the evidence is thin or contradictory
- key_factors: 3-5 brief bullet strings, most important first
"""


def extract_kalshi(
    report_markdown: str,
    market_question: str,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """
    Extract a binary probability for a Kalshi prediction market.

    Args:
        report_markdown: The full_report.md content
        market_question: The exact market question e.g.
            "Will the Federal Reserve cut rates by 50bp in September 2025?"
        llm_client: Optional pre-initialised LLMClient (defaults to new instance)

    Returns:
        Dict with keys: yes_probability, no_probability, confidence,
                        key_factors, reasoning_summary
    """
    client = llm_client or LLMClient()

    user_content = (
        f"PREDICTION MARKET QUESTION:\n{market_question}\n\n"
        f"RESEARCH REPORT:\n{report_markdown[:8000]}"
    )

    messages = [
        {"role": "system", "content": KALSHI_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        result = client.chat_json(messages, temperature=0.1, max_tokens=1024)
        _validate_kalshi(result)
        logger.info(f"Kalshi extraction: yes={result.get('yes_probability')}, conf={result.get('confidence')}")
        return result
    except Exception as e:
        logger.error(f"Kalshi extraction failed: {e}")
        return {
            "yes_probability": 0.5,
            "no_probability": 0.5,
            "confidence": "low",
            "key_factors": ["Extraction failed — defaulting to 50/50"],
            "reasoning_summary": f"Probability extraction failed: {e}",
        }


def _validate_kalshi(result: Dict) -> None:
    yes = result.get("yes_probability", 0)
    no = result.get("no_probability", 0)
    total = round(yes + no, 6)
    if abs(total - 1.0) > 0.01:
        # Auto-normalise
        if yes + no > 0:
            result["yes_probability"] = round(yes / (yes + no), 4)
            result["no_probability"] = round(no / (yes + no), 4)
        else:
            result["yes_probability"] = 0.5
            result["no_probability"] = 0.5


# ---------------------------------------------------------------------------
# Sports probability
# ---------------------------------------------------------------------------

SPORTS_SYSTEM_PROMPT = """You are a professional sports betting analyst with a track record of calibrated predictions.
You will read an AI-generated research report about an upcoming sports game and extract
structured probability estimates for betting markets.

CRITICAL PROCESS — follow these steps IN ORDER before outputting JSON:
1. MARKET ANCHOR: Find the BETTING ODDS section. Convert the odds to vig-free implied
   win probabilities (normalise the raw implied probs to remove bookmaker margin).
   This is your single most important anchor — betting markets are highly efficient.
2. STATISTICAL PRIOR: Find the ML STATISTICAL PREDICTION section. Note the XGBoost
   model's moneyline probability. Weight this heavily alongside the market line.
3. INJURY ADJUSTMENT: Find the INJURY REPORT section. Missing star players (Out/Doubtful)
   are the most under-priced factor. Adjust by 5-12 percentage points per impactful absence.
4. SIMULATION EVIDENCE: Use the simulation agents' consensus to make smaller fine-tuning
   adjustments (±3-5 percentage points maximum).
5. FINAL RULE: Do NOT drift more than 12 percentage points from the market-implied
   probability without a specific, named injury or situational factor justifying it.
   Markets price most public information correctly.

Return ONLY a valid JSON object — no markdown, no explanations outside the JSON.

Required JSON shape:
{
  "moneyline": {
    "team_a": "<team name>",
    "team_a_probability": <float 0.0-1.0>,
    "team_b_probability": <float 0.0-1.0>,
    "market_implied_team_a": <float 0.0-1.0 or null if odds unavailable>,
    "ml_model_team_a": <float 0.0-1.0 or null if ML section absent>
  },
  "spread": {
    "line": <float, e.g. -3.5>,
    "favorite": "<team name>",
    "cover_probability": <float 0.0-1.0>
  },
  "total": {
    "line": <float, e.g. 220.5>,
    "over_probability": <float 0.0-1.0>
  },
  "player_props": [
    {
      "player": "<player name>",
      "market": "<points|rebounds|assists|goals|etc>",
      "line": <float>,
      "over_probability": <float 0.0-1.0>
    }
  ],
  "confidence": "<low|medium|high>",
  "reasoning_summary": "<3-5 sentence justification: state market line used, ML model output, key injury/situational adjustments, and final rationale>"
}

Rules:
- moneyline: team_a_probability + team_b_probability must equal 1.0
- spread.cover_probability is the probability the favourite covers the spread
- total.over_probability is the probability the game goes over the total line
- player_props can be an empty array [] if no prop data is available
- If a market cannot be estimated, set its probability to 0.5
- confidence = "high" only when: market line and ML model agree within 5% AND no key injuries AND simulation is consistent
- confidence = "medium" if 1-2 meaningful uncertainties exist
- confidence = "low" if injuries, model disagreement, or line movement creates >10% uncertainty
"""


def extract_sports(
    report_markdown: str,
    sport_config,
    llm_client: Optional[LLMClient] = None,
    ml_prediction: Optional[Dict[str, Any]] = None,
    n_ensemble: int = 1,
) -> Dict[str, Any]:
    """
    Extract structured betting probabilities from a sports report.

    Makes n_ensemble LLM calls and averages the results, then blends with the
    ML model output for a calibrated final estimate.

    n_ensemble defaults to 1 — a single call is already well-calibrated when
    blended with the ML model (35% ML / 65% LLM).  Raise to 2-3 only when you
    need higher variance reduction and cost is not a concern.

    Args:
        report_markdown: The full_report.md content
        sport_config: SportConfig dataclass or dict with team/bet context
        llm_client: Optional pre-initialised LLMClient
        ml_prediction: Optional ML model output dict (from ml_prediction_service).
                       When provided, blended with LLM ensemble at 35/65 weight.
        n_ensemble: Number of independent LLM extractions to average (default 1).

    Returns:
        Dict with keys: moneyline, spread, total, player_props, confidence,
                        reasoning_summary
    """
    client = llm_client or LLMClient()

    # Build context header
    if hasattr(sport_config, "team_a_name"):
        team_a = sport_config.team_a_name
        team_b = sport_config.team_b_name
        bet_types = sport_config.bet_types or []
        prop_players = sport_config.player_prop_players or []
    else:
        team_a = sport_config.get("team_a_name", "Team A")
        team_b = sport_config.get("team_b_name", "Team B")
        bet_types = sport_config.get("bet_types", [])
        prop_players = sport_config.get("player_prop_players", [])

    context = (
        f"MATCHUP: {team_a} vs {team_b}\n"
        f"BET TYPES TO ESTIMATE: {', '.join(bet_types) if bet_types else 'moneyline, spread, total'}\n"
    )
    if prop_players:
        context += f"PLAYER PROPS FOR: {', '.join(prop_players)}\n"

    user_content = f"{context}\nRESEARCH REPORT:\n{report_markdown[:12000]}"

    messages = [
        {"role": "system", "content": SPORTS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    default = _default_sports_result(team_a, team_b)

    # --- Ensemble: run n_ensemble extractions and average ---
    successful_results: List[Dict] = []
    temperatures = [0.1, 0.2, 0.15][:n_ensemble]  # slight variation for diversity

    for i, temp in enumerate(temperatures):
        try:
            r = client.chat_json(messages, temperature=temp, max_tokens=1500)
            _validate_sports(r, team_a, team_b)
            successful_results.append(r)
        except Exception as e:
            logger.warning(f"Sports extraction attempt {i+1}/{n_ensemble} failed: {e}")

    if not successful_results:
        logger.error("All sports extraction attempts failed — returning default")
        default["reasoning_summary"] = "All probability extraction attempts failed."
        return default

    # Average probabilities across ensemble
    result = _average_sports_ensemble(successful_results, team_a, team_b)

    # --- Bayesian blend with ML model output (35% ML, 65% LLM ensemble) ---
    if ml_prediction:
        ml_prob = ml_prediction.get("moneyline_prob")  # home team win prob
        if ml_prob is not None:
            lm_a = result["moneyline"]["team_a_probability"]
            blended_a = round(0.35 * float(ml_prob) + 0.65 * lm_a, 4)
            blended_b = round(1.0 - blended_a, 4)
            result["moneyline"]["team_a_probability"] = blended_a
            result["moneyline"]["team_b_probability"] = blended_b
            result["moneyline"]["ml_model_team_a"] = round(float(ml_prob), 4)
            logger.info(f"ML blend applied: ml={ml_prob:.3f} + llm_ensemble={lm_a:.3f} → {blended_a:.3f}")

    logger.info(
        f"Sports extraction (ensemble={len(successful_results)}): "
        f"{team_a} p={result['moneyline']['team_a_probability']}, "
        f"conf={result.get('confidence')}"
    )
    return result


def _average_sports_ensemble(results: List[Dict], team_a: str, team_b: str) -> Dict[str, Any]:
    """Average probabilities across multiple extraction results."""
    if len(results) == 1:
        return results[0]

    base = results[0]

    # Average moneyline
    a_probs = [r.get("moneyline", {}).get("team_a_probability", 0.5) for r in results]
    avg_a = round(sum(a_probs) / len(a_probs), 4)
    base["moneyline"]["team_a_probability"] = avg_a
    base["moneyline"]["team_b_probability"] = round(1.0 - avg_a, 4)

    # Average spread cover probability
    cover_probs = [r.get("spread", {}).get("cover_probability", 0.5) for r in results]
    base.setdefault("spread", {})["cover_probability"] = round(sum(cover_probs) / len(cover_probs), 4)

    # Average total over probability
    over_probs = [r.get("total", {}).get("over_probability", 0.5) for r in results]
    base.setdefault("total", {})["over_probability"] = round(sum(over_probs) / len(over_probs), 4)

    # Confidence: take the most conservative (lowest confidence wins)
    conf_rank = {"high": 2, "medium": 1, "low": 0}
    confs = [r.get("confidence", "low") for r in results]
    min_conf = min(confs, key=lambda c: conf_rank.get(c, 0))
    base["confidence"] = min_conf

    return base


def _validate_sports(result: Dict, team_a: str, team_b: str) -> None:
    ml = result.get("moneyline", {})
    a_prob = ml.get("team_a_probability", 0)
    b_prob = ml.get("team_b_probability", 0)
    if abs(a_prob + b_prob - 1.0) > 0.01:
        total = a_prob + b_prob
        if total > 0:
            result["moneyline"]["team_a_probability"] = round(a_prob / total, 4)
            result["moneyline"]["team_b_probability"] = round(b_prob / total, 4)
        else:
            result["moneyline"]["team_a_probability"] = 0.5
            result["moneyline"]["team_b_probability"] = 0.5

    if "team_a" not in ml:
        result.setdefault("moneyline", {})["team_a"] = team_a
    if "player_props" not in result:
        result["player_props"] = []


def _default_sports_result(team_a: str, team_b: str) -> Dict[str, Any]:
    return {
        "moneyline": {
            "team_a": team_a,
            "team_a_probability": 0.5,
            "team_b_probability": 0.5,
        },
        "spread": {"line": 0.0, "favorite": team_a, "cover_probability": 0.5},
        "total": {"line": 0.0, "over_probability": 0.5},
        "player_props": [],
        "confidence": "low",
        "reasoning_summary": "Default 50/50 — extraction unavailable.",
    }
