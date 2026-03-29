"""
Polymarket Predictor

Extends KalshiPredictor calibration logic to Polymarket markets.

Key insight: Polymarket markets follow similar patterns to Kalshi:
  - Binary YES/NO outcomes
  - Well-calibrated crowd probability
  - Edge through base rate analysis + economic indicators
  - Similar Kelly sizing applies

This predictor:
  1. Takes Polymarket market question + current price
  2. Applies category-specific calibration
  3. Computes edge vs market price
  4. Outputs order signal (BUY YES / BUY NO / NEUTRAL)
  5. Suggests Kelly position sizing
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

from .kalshi_predictor import KalshiPredictor, CATEGORY_BASE_RATES

logger = logging.getLogger("mirofish.ml.polymarket")


class PolymarketPredictor:
    """
    Prediction model for Polymarket binary markets.

    Reuses KalshiPredictor's calibration engine, but adapted for:
      - Polymarket market structure
      - Lower liquidity on some markets (affects confidence)
      - CLOB-based pricing (different from AMM)
    """

    def __init__(self):
        # Use KalshiPredictor as backbone for calibration
        self.kalshi_predictor = KalshiPredictor()
        self.is_trained = False
        self.liquidity_threshold = 1000  # USD min volume for high confidence

    def predict(self,
                market_id: str,
                question: str,
                yes_price: float,
                no_price: float,
                volume: float = 0,
                category: str = "unknown",
                time_to_close_days: float = 30.0,
                economic_context: Optional[Dict] = None,
                ) -> Dict[str, Any]:
        """
        Predict YES/NO probability for a Polymarket market.

        Args:
            market_id: Polymarket market ID
            question: Full market question text
            yes_price: Current YES price (bid/ask midpoint or last trade)
            no_price: Current NO price
            volume: 24h trading volume in USDC
            category: "politics", "economics", "sports", "crypto", "science", etc.
            time_to_close_days: Days until market resolves
            economic_context: Dict with FRED data (CPI, Fed rates, etc.) for econ questions

        Returns:
            {
                "market_id": str,
                "question": str,
                "yes_probability": float — calibrated YES probability
                "no_probability": float — calibrated NO probability
                "market_yes_price": float — observed market price
                "market_no_price": float — observed market price
                "edge": float — difference between model and market
                "edge_signal": str — "BUY YES", "BUY NO", "NEUTRAL"
                "kelly_fraction": float — fractional Kelly (typically 0.25)
                "suggested_order_size": float — size in USDC
                "liquidity_score": float — 0-100, higher = better
                "confidence": str — "high", "medium", "low"
                "factors": List[str] — reasoning
                "reasoning_summary": str — brief explanation for trading
            }
        """
        # Ensure Kalshi predictor is trained
        if not self.kalshi_predictor.is_trained:
            if not self.kalshi_predictor.load():
                self.kalshi_predictor._build_fallback_calibrator()

        # Validate prices
        yes_price = max(0.01, min(0.99, yes_price))
        no_price = max(0.01, min(0.99, no_price))

        # Sanity check: YES + NO should be close to 1.0 (minus spreads/fees)
        price_sum = yes_price + no_price
        if abs(price_sum - 1.0) > 0.10:
            logger.warning(f"Unusual price sum {price_sum} for {market_id}")

        # Use YES price as the primary market probability
        market_yes_prob = yes_price

        # Use KalshiPredictor's calibration logic
        kalshi_pred = self.kalshi_predictor.predict(
            market_question=question,
            community_prob=market_yes_prob,
            category=category,
            time_to_close_days=time_to_close_days,
            economic_context=economic_context,
        )

        # Extract calibrated probability
        calibrated_yes_prob = kalshi_pred["yes_probability"]

        # Calculate edge
        edge = calibrated_yes_prob - market_yes_prob

        # Compute liquidity score (0-100)
        liquidity = self._compute_liquidity_score(volume)

        # Confidence adjusted for liquidity
        confidence = kalshi_pred["confidence"]
        if volume < self.liquidity_threshold:
            # Low liquidity = harder to execute, reduce confidence
            if confidence == "high":
                confidence = "medium"
            elif confidence == "medium":
                confidence = "low"

        # Edge signal with minimum edge threshold
        edge_threshold = 0.04  # 4% minimum edge to trade
        if abs(edge) < edge_threshold:
            edge_signal = "NEUTRAL"
            order_size = 0
        elif edge > edge_threshold:
            edge_signal = "BUY YES"
            order_size = self._compute_order_size(
                edge=edge,
                confidence=confidence,
                liquidity=liquidity,
                max_risk=100,  # Max $100 per trade
            )
        else:  # edge < -edge_threshold
            edge_signal = "BUY NO"
            order_size = self._compute_order_size(
                edge=abs(edge),
                confidence=confidence,
                liquidity=liquidity,
                max_risk=100,
            )

        # Kelly fraction (from Kalshi predictor)
        kelly = kalshi_pred["kelly_fraction"]

        # Build factors explanation
        factors = kalshi_pred.get("factors", [])
        factors.append(f"Polymarket YES price: {market_yes_prob:.1%}")
        factors.append(f"Market liquidity: {liquidity:.0f}/100")

        if edge > 0:
            factors.append(f"Edge: Model higher than market by {edge:.1%}")
        elif edge < 0:
            factors.append(f"Edge: Model lower than market by {abs(edge):.1%}")

        # Reasoning summary
        if edge_signal == "NEUTRAL":
            summary = "No significant edge detected."
        elif edge_signal == "BUY YES":
            summary = (f"Model assigns {calibrated_yes_prob:.0%} YES vs market {market_yes_prob:.0%}. "
                      f"Buy YES at {market_yes_prob:.1%}, target {calibrated_yes_prob:.0%}.")
        else:  # BUY NO
            calibrated_no_prob = 1 - calibrated_yes_prob
            market_no_prob = 1 - market_yes_prob
            summary = (f"Model assigns {calibrated_no_prob:.0%} NO vs market {market_no_prob:.0%}. "
                      f"Buy NO at {market_no_prob:.1%}, target {calibrated_no_prob:.0%}.")

        return {
            "market_id": market_id,
            "question": question,
            "yes_probability": round(calibrated_yes_prob, 4),
            "no_probability": round(1 - calibrated_yes_prob, 4),
            "market_yes_price": round(market_yes_prob, 4),
            "market_no_price": round(1 - market_yes_prob, 4),
            "edge": round(edge, 4),
            "edge_signal": edge_signal,
            "kelly_fraction": round(kelly, 3),
            "suggested_order_size": round(order_size, 2),
            "liquidity_score": round(liquidity, 1),
            "confidence": confidence,
            "factors": factors,
            "reasoning_summary": summary,
            "volume_24h": volume,
            "time_to_close_days": time_to_close_days,
        }

    def predict_batch(self, markets: list) -> list:
        """
        Predict on multiple markets at once.

        Input: List of dicts with keys: market_id, question, yes_price, no_price, volume, category, ...
        Output: List of predictions
        """
        predictions = []
        for market in markets:
            try:
                pred = self.predict(
                    market_id=market.get("id"),
                    question=market.get("question"),
                    yes_price=market.get("yes_price", 0.5),
                    no_price=market.get("no_price", 0.5),
                    volume=market.get("volume", 0),
                    category=market.get("category", "unknown"),
                    time_to_close_days=market.get("days_to_close", 30),
                )
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Prediction failed for {market.get('id')}: {e}")
        return predictions

    # =========================================================================
    # Helpers
    # =========================================================================

    def _compute_liquidity_score(self, volume: float) -> float:
        """
        Compute liquidity score 0-100.

        0-1k: 0-20 (illiquid, high slippage risk)
        1k-5k: 20-50 (low liquidity)
        5k-20k: 50-80 (moderate liquidity)
        20k+: 80-100 (good liquidity)
        """
        if volume < 1000:
            return (volume / 1000) * 20
        elif volume < 5000:
            return 20 + (volume - 1000) / 4000 * 30
        elif volume < 20000:
            return 50 + (volume - 5000) / 15000 * 30
        else:
            return min(100, 80 + (volume - 20000) / 100000 * 20)

    def _compute_order_size(self,
                           edge: float,
                           confidence: str,
                           liquidity: float,
                           max_risk: float = 100) -> float:
        """
        Compute order size based on edge, confidence, and liquidity.

        General logic:
          - Larger edge = larger position
          - Higher confidence = larger position
          - Better liquidity = can go larger
          - Clipped to max_risk
        """
        base_size = max_risk

        # Edge multiplier (higher edge = bigger position)
        edge_multiplier = min(3.0, 1.0 + edge / 0.02)  # Each 2% edge adds 50%

        # Confidence multiplier
        conf_mult = {
            "high": 1.0,
            "medium": 0.7,
            "low": 0.4,
        }.get(confidence, 0.5)

        # Liquidity multiplier (avoid over-sizing in thin markets)
        liquidity_mult = min(1.0, liquidity / 50)  # 50/100 liquidity = full size

        # Compute final size
        size = base_size * edge_multiplier * conf_mult * liquidity_mult

        return min(size, max_risk)

    def health_check(self) -> Dict[str, Any]:
        """Check predictor health."""
        return {
            "kalshi_trained": self.kalshi_predictor.is_trained,
            "model_loaded": self.kalshi_predictor.is_trained,
            "timestamp": str(np.datetime64('now')),
        }
