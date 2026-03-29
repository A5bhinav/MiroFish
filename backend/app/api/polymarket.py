"""
Polymarket API Blueprint

Endpoints:
  GET    /api/polymarket/catalog          — list all active markets
  GET    /api/polymarket/search?q=...      — search for markets
  GET    /api/polymarket/markets/{id}      — fetch single market details
  POST   /api/polymarket/predict           — get prediction for a market
  POST   /api/polymarket/predict/batch     — predict on multiple markets
  POST   /api/polymarket/trade             — place a trade
  GET    /api/polymarket/orders            — get active orders
  GET    /api/polymarket/positions         — get current positions
  GET    /api/polymarket/status            — executor health check
"""

import traceback
import threading
from flask import request, jsonify, Blueprint

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..utils.logger import get_logger
from ..services.polymarket_data_fetcher import PolymarketDataFetcher
from ..ml.polymarket_predictor import PolymarketPredictor

logger = get_logger("mirofish.polymarket_api")

# Create blueprint
polymarket_bp = Blueprint("polymarket", __name__, url_prefix="/api/polymarket")

# Global instances
data_fetcher = PolymarketDataFetcher()
predictor = PolymarketPredictor()


# ---------------------------------------------------------------------------
# Market Discovery
# ---------------------------------------------------------------------------

@polymarket_bp.route("/catalog", methods=["GET"])
def get_catalog():
    """List all active Polymarket markets."""
    try:
        limit = request.args.get("limit", 100, type=int)
        offset = request.args.get("offset", 0, type=int)

        markets = data_fetcher.get_active_markets(limit=limit, offset=offset)
        return jsonify({
            "success": True,
            "data": {
                "markets": markets,
                "count": len(markets),
                "limit": limit,
                "offset": offset,
            }
        })
    except Exception as e:
        logger.error(f"Catalog fetch failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@polymarket_bp.route("/search", methods=["GET"])
def search_markets():
    """Search for markets by keyword."""
    try:
        query = request.args.get("q", "")
        if not query:
            return jsonify({"success": False, "error": "Missing query parameter 'q'"}), 400

        limit = request.args.get("limit", 20, type=int)
        markets = data_fetcher.search_markets(query, limit=limit)

        return jsonify({
            "success": True,
            "data": {
                "query": query,
                "markets": markets,
                "count": len(markets),
            }
        })
    except Exception as e:
        logger.error(f"Market search failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@polymarket_bp.route("/category/<category>", methods=["GET"])
def get_by_category(category):
    """List markets in a category."""
    try:
        limit = request.args.get("limit", 50, type=int)
        markets = data_fetcher.get_markets_by_category(category, limit=limit)

        return jsonify({
            "success": True,
            "data": {
                "category": category,
                "markets": markets,
                "count": len(markets),
            }
        })
    except Exception as e:
        logger.error(f"Category fetch failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@polymarket_bp.route("/markets/<market_id>", methods=["GET"])
def get_market(market_id):
    """Get detailed info for a single market."""
    try:
        market = data_fetcher.get_market_metadata(market_id)
        if not market:
            return jsonify({"success": False, "error": f"Market {market_id} not found"}), 404

        # Add current pricing
        pricing = data_fetcher.get_current_price(market_id)
        if pricing:
            market.update(pricing)

        return jsonify({
            "success": True,
            "data": market
        })
    except Exception as e:
        logger.error(f"Market fetch failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

@polymarket_bp.route("/predict", methods=["POST"])
def predict():
    """Get prediction for a Polymarket market."""
    try:
        body = request.get_json(force=True) or {}

        # Required fields
        market_id = body.get("market_id")
        question = body.get("question")
        yes_price = body.get("yes_price", 0.5)

        if not market_id or not question:
            return jsonify({
                "success": False,
                "error": "Missing required fields: market_id, question"
            }), 400

        # Optional fields
        no_price = body.get("no_price", 1 - yes_price)
        volume = body.get("volume", 0)
        category = body.get("category", "unknown")
        time_to_close_days = body.get("days_to_close", 30)

        # Run prediction
        prediction = predictor.predict(
            market_id=market_id,
            question=question,
            yes_price=yes_price,
            no_price=no_price,
            volume=volume,
            category=category,
            time_to_close_days=time_to_close_days,
        )

        return jsonify({
            "success": True,
            "data": prediction
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@polymarket_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Get predictions for multiple markets."""
    try:
        body = request.get_json(force=True) or {}
        markets = body.get("markets", [])

        if not markets:
            return jsonify({"success": False, "error": "Missing 'markets' array"}), 400

        predictions = predictor.predict_batch(markets)

        return jsonify({
            "success": True,
            "data": {
                "count": len(predictions),
                "predictions": predictions,
            }
        })

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@polymarket_bp.route("/scan-opportunities", methods=["GET"])
def scan_opportunities():
    """Scan all markets for trading opportunities (top edges)."""
    try:
        # Fetch high-liquidity markets
        markets = data_fetcher.filter_high_liquidity_markets(min_volume=1000, min_traders=5)

        # Run predictor on each
        opportunities = predictor.predict_batch(markets)

        # Sort by edge (largest first)
        opportunities.sort(key=lambda x: abs(x.get("edge", 0)), reverse=True)

        # Filter to only those with edge_signal != NEUTRAL
        actionable = [p for p in opportunities if p.get("edge_signal") != "NEUTRAL"]

        return jsonify({
            "success": True,
            "data": {
                "opportunities": actionable[:20],  # Top 20
                "total_scanned": len(markets),
                "actionable_count": len(actionable),
            }
        })

    except Exception as e:
        logger.error(f"Opportunity scan failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Trading (Execution)
# ---------------------------------------------------------------------------

@polymarket_bp.route("/trade", methods=["POST"])
def place_trade():
    """Place a trade on Polymarket (requires authentication)."""
    try:
        body = request.get_json(force=True) or {}

        # Check authorization (would be done via API key in production)
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        if api_key != Config.POLYMARKET_API_KEY and Config.POLYMARKET_API_KEY:
            return jsonify({"success": False, "error": "Unauthorized"}), 401

        # Import executor (lazy load to avoid requiring py-clob-client in all cases)
        from ..services.polymarket_executor import PolymarketExecutor

        executor = PolymarketExecutor(
            private_key=Config.POLYMARKET_PRIVATE_KEY,
            api_key=Config.POLYMARKET_API_KEY,
            mode=Config.POLYMARKET_MODE,
            dry_run=Config.POLYMARKET_DRY_RUN,
        )

        # Parse order
        market_id = body.get("market_id")
        side = body.get("side")  # "BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO"
        amount = body.get("amount")  # USDC amount
        price = body.get("price")  # Limit price [0, 1]
        kelly_fraction = body.get("kelly_fraction", 0.25)

        if not all([market_id, side, amount, price]):
            return jsonify({
                "success": False,
                "error": "Missing required fields: market_id, side, amount, price"
            }), 400

        # Place order
        order = executor.place_order(
            market_id=market_id,
            side=side,
            amount=amount,
            price=price,
            kelly_fraction=kelly_fraction,
        )

        if not order:
            return jsonify({"success": False, "error": "Order placement failed"}), 500

        return jsonify({
            "success": True,
            "data": order
        })

    except Exception as e:
        logger.error(f"Trade placement failed: {e}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@polymarket_bp.route("/orders", methods=["GET"])
def get_orders():
    """Get active orders."""
    try:
        from ..services.polymarket_executor import PolymarketExecutor

        executor = PolymarketExecutor(
            private_key=Config.POLYMARKET_PRIVATE_KEY,
            api_key=Config.POLYMARKET_API_KEY,
        )

        orders = executor.get_active_orders()

        return jsonify({
            "success": True,
            "data": {
                "orders": orders,
                "count": len(orders),
            }
        })

    except Exception as e:
        logger.error(f"Failed to fetch orders: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@polymarket_bp.route("/positions", methods=["GET"])
def get_positions():
    """Get current open positions."""
    try:
        from ..services.polymarket_executor import PolymarketExecutor

        executor = PolymarketExecutor(
            private_key=Config.POLYMARKET_PRIVATE_KEY,
            api_key=Config.POLYMARKET_API_KEY,
        )

        positions = executor.get_positions()

        return jsonify({
            "success": True,
            "data": {
                "positions": positions,
                "count": len(positions),
            }
        })

    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Status & Health
# ---------------------------------------------------------------------------

@polymarket_bp.route("/status", methods=["GET"])
def get_status():
    """Get executor health status."""
    try:
        from ..services.polymarket_executor import PolymarketExecutor

        executor = PolymarketExecutor(
            private_key=Config.POLYMARKET_PRIVATE_KEY,
            api_key=Config.POLYMARKET_API_KEY,
        )

        health = executor.health_check()
        predictor_health = predictor.health_check()

        return jsonify({
            "success": True,
            "data": {
                "executor": health,
                "predictor": predictor_health,
            }
        })

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
