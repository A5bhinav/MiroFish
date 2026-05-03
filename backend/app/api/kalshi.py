"""
Kalshi API Blueprint

Endpoints:
  GET  /api/kalshi/catalog                 — list active markets
  GET  /api/kalshi/search?q=...            — keyword search
  GET  /api/kalshi/category/<cat>          — markets by category
  GET  /api/kalshi/markets/<ticker>        — single market detail + live price
  POST /api/kalshi/predict                 — predict with live market price
  POST /api/kalshi/predict/batch           — predict multiple markets
  GET  /api/kalshi/scan-opportunities      — scan for edge across active markets
  POST /api/kalshi/trade                   — place a trade (requires credentials)
  GET  /api/kalshi/orders                  — active orders
  GET  /api/kalshi/positions               — open positions
  GET  /api/kalshi/balance                 — account balance
  GET  /api/kalshi/status                  — health check
"""

import traceback
from flask import request, jsonify, Blueprint

from ..config import Config
from ..utils.logger import get_logger
from ..services.kalshi_data_fetcher import KalshiDataFetcher
from ..ml.kalshi_predictor import KalshiPredictor

logger = get_logger("mirofish.kalshi_api")

kalshi_bp = Blueprint("kalshi", __name__, url_prefix="/api/kalshi")

# Module-level singletons (lazy-initialise the predictor on first use)
data_fetcher = KalshiDataFetcher()
_predictor: KalshiPredictor = None


def _get_predictor() -> KalshiPredictor:
    global _predictor
    if _predictor is None:
        _predictor = KalshiPredictor()
        if not _predictor.load():
            _predictor._build_fallback_calibrator()
    return _predictor


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------

@kalshi_bp.route("/catalog", methods=["GET"])
def get_catalog():
    """List active Kalshi markets."""
    try:
        limit  = request.args.get("limit", 100, type=int)
        cursor = request.args.get("cursor", "")
        markets = data_fetcher.get_active_markets(limit=limit, cursor=cursor)
        return jsonify({
            "success": True,
            "data": {
                "markets": markets,
                "count":   len(markets),
                "limit":   limit,
            }
        })
    except Exception as e:
        logger.error(f"Kalshi catalog fetch failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@kalshi_bp.route("/search", methods=["GET"])
def search_markets():
    """Search markets by keyword (client-side filter)."""
    try:
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify({"success": False, "error": "Missing query parameter 'q'"}), 400

        limit   = request.args.get("limit", 20, type=int)
        markets = data_fetcher.search_markets(query, limit=limit)
        return jsonify({
            "success": True,
            "data": {
                "query":   query,
                "markets": markets,
                "count":   len(markets),
            }
        })
    except Exception as e:
        logger.error(f"Kalshi search failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@kalshi_bp.route("/category/<category>", methods=["GET"])
def get_by_category(category):
    """List markets in a Kalshi category (e.g. 'economics', 'politics')."""
    try:
        limit   = request.args.get("limit", 50, type=int)
        markets = data_fetcher.get_markets_by_category(category, limit=limit)
        return jsonify({
            "success": True,
            "data": {
                "category": category,
                "markets":  markets,
                "count":    len(markets),
            }
        })
    except Exception as e:
        logger.error(f"Kalshi category fetch failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@kalshi_bp.route("/markets/<ticker>", methods=["GET"])
def get_market(ticker):
    """Fetch detail and live price for a single market."""
    try:
        market = data_fetcher.get_market_metadata(ticker)
        if not market:
            return jsonify({"success": False, "error": f"Market '{ticker}' not found"}), 404

        pricing = data_fetcher.get_current_price(ticker)
        if pricing:
            market.update(pricing)

        return jsonify({"success": True, "data": market})
    except Exception as e:
        logger.error(f"Kalshi market fetch failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

@kalshi_bp.route("/predict", methods=["POST"])
def predict():
    """
    Predict a Kalshi market.

    If 'ticker' is supplied, live price is fetched automatically.
    Otherwise pass 'yes_price' manually (0–1 probability).

    Body:
      {
        "ticker":         "BTCUSD-24DEC31",   // optional if yes_price provided
        "question":       "Will BTC close...", // required if no ticker
        "yes_price":      0.45,                // optional — overridden by live price
        "category":       "crypto",
        "days_to_close":  30
      }
    """
    try:
        body     = request.get_json(force=True) or {}
        ticker   = body.get("ticker", "").strip()
        question = body.get("question", "").strip()
        yes_price = float(body.get("yes_price", 0.5))
        category  = body.get("category", "economics")
        days_to_close = float(body.get("days_to_close", 30))

        # Auto-fetch live price when ticker is given
        if ticker:
            market = data_fetcher.get_market(ticker)
            if market:
                yes_price     = market["yes_price"]
                days_to_close = market.get("days_to_close", days_to_close)
                category      = market.get("category", category)
                if not question:
                    question = market.get("question", ticker)

        if not question:
            return jsonify({"success": False, "error": "Provide 'ticker' or 'question'"}), 400

        predictor = _get_predictor()
        econ_ctx  = predictor.get_economic_context()
        prediction = predictor.predict(
            market_question=question,
            community_prob=yes_price,
            category=category,
            time_to_close_days=days_to_close,
            economic_context=econ_ctx,
        )

        return jsonify({
            "success": True,
            "data": {
                **prediction,
                "ticker":        ticker or None,
                "question":      question,
                "live_yes_price": yes_price,
                "source":         "kalshi_live" if ticker else "manual",
            }
        })

    except Exception as e:
        logger.error(f"Kalshi predict failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@kalshi_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict multiple Kalshi markets.

    Body:
      {
        "markets": [
          {"ticker": "...", "question": "...", "yes_price": 0.5, "category": "..."},
          ...
        ]
      }
    """
    try:
        body    = request.get_json(force=True) or {}
        markets = body.get("markets", [])
        if not markets:
            return jsonify({"success": False, "error": "Missing 'markets' array"}), 400

        predictor = _get_predictor()
        econ_ctx  = predictor.get_economic_context()
        results   = []

        for m in markets:
            ticker    = m.get("ticker", "")
            question  = m.get("question", "")
            yes_price = float(m.get("yes_price", 0.5))
            category  = m.get("category", "economics")
            days_cls  = float(m.get("days_to_close", 30))

            # Auto-fetch live price when ticker given and yes_price not explicit
            if ticker and "yes_price" not in m:
                live = data_fetcher.get_market(ticker)
                if live:
                    yes_price = live["yes_price"]
                    days_cls  = live.get("days_to_close", days_cls)
                    category  = live.get("category", category)
                    if not question:
                        question = live.get("question", ticker)

            if not question:
                continue

            try:
                pred = predictor.predict(
                    market_question=question,
                    community_prob=yes_price,
                    category=category,
                    time_to_close_days=days_cls,
                    economic_context=econ_ctx,
                )
                results.append({**pred, "ticker": ticker, "question": question,
                                 "live_yes_price": yes_price})
            except Exception as e:
                results.append({"ticker": ticker, "question": question, "error": str(e)})

        return jsonify({
            "success": True,
            "data": {"count": len(results), "predictions": results}
        })

    except Exception as e:
        logger.error(f"Kalshi batch predict failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@kalshi_bp.route("/scan-opportunities", methods=["GET"])
def scan_opportunities():
    """
    Scan active markets for trading edge.
    Returns the top 20 markets sorted by |predicted_prob − market_price|.
    """
    try:
        predictor    = _get_predictor()
        opportunities = data_fetcher.find_mispriced_markets(predictor)
        actionable    = [o for o in opportunities if o.get("edge_signal") != "NEUTRAL"]

        return jsonify({
            "success": True,
            "data": {
                "opportunities":   actionable[:20],
                "total_scanned":   len(opportunities),
                "actionable_count": len(actionable),
            }
        })

    except Exception as e:
        logger.error(f"Kalshi opportunity scan failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------

@kalshi_bp.route("/trade", methods=["POST"])
def place_trade():
    """
    Place an order on Kalshi.  Requires KALSHI_API_KEY and KALSHI_PRIVATE_KEY.

    Body:
      {
        "ticker":         "BTCUSD-24DEC31",
        "side":           "BUY_YES",          // BUY_YES | BUY_NO | SELL_YES | SELL_NO
        "amount":         50.0,               // USD to risk (before Kelly sizing)
        "price":          0.45,               // limit price as probability
        "kelly_fraction": 0.25                // optional, default 0.25
      }
    """
    try:
        body           = request.get_json(force=True) or {}
        ticker         = body.get("ticker")
        side           = body.get("side")
        amount         = body.get("amount")
        price          = body.get("price")
        kelly_fraction = float(body.get("kelly_fraction", 0.25))

        if not all([ticker, side, amount, price]):
            return jsonify({
                "success": False,
                "error":   "Missing required fields: ticker, side, amount, price"
            }), 400

        from ..services.kalshi_executor import KalshiExecutor
        executor = KalshiExecutor(
            api_key     = getattr(Config, "KALSHI_API_KEY",     None),
            private_key = getattr(Config, "KALSHI_PRIVATE_KEY", None),
            dry_run     = getattr(Config, "KALSHI_DRY_RUN",     True),
        )

        order = executor.place_order(
            ticker         = ticker,
            side           = side,
            amount         = float(amount),
            price          = float(price),
            kelly_fraction = kelly_fraction,
        )

        if not order:
            return jsonify({"success": False, "error": "Order placement failed"}), 500

        return jsonify({"success": True, "data": order})

    except Exception as e:
        logger.error(f"Kalshi trade failed: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@kalshi_bp.route("/orders", methods=["GET"])
def get_orders():
    """Get active/resting orders."""
    try:
        from ..services.kalshi_executor import KalshiExecutor
        executor = KalshiExecutor(
            api_key     = getattr(Config, "KALSHI_API_KEY",     None),
            private_key = getattr(Config, "KALSHI_PRIVATE_KEY", None),
            dry_run     = getattr(Config, "KALSHI_DRY_RUN",     True),
        )
        orders = executor.get_active_orders()
        return jsonify({"success": True, "data": {"orders": orders, "count": len(orders)}})
    except Exception as e:
        logger.error(f"Kalshi orders fetch failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@kalshi_bp.route("/positions", methods=["GET"])
def get_positions():
    """Get open positions."""
    try:
        ticker = request.args.get("ticker")
        from ..services.kalshi_executor import KalshiExecutor
        executor = KalshiExecutor(
            api_key     = getattr(Config, "KALSHI_API_KEY",     None),
            private_key = getattr(Config, "KALSHI_PRIVATE_KEY", None),
            dry_run     = getattr(Config, "KALSHI_DRY_RUN",     True),
        )
        positions = executor.get_positions(ticker=ticker)
        return jsonify({"success": True, "data": {"positions": positions, "count": len(positions)}})
    except Exception as e:
        logger.error(f"Kalshi positions fetch failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@kalshi_bp.route("/balance", methods=["GET"])
def get_balance():
    """Get account cash balance."""
    try:
        from ..services.kalshi_executor import KalshiExecutor
        executor = KalshiExecutor(
            api_key     = getattr(Config, "KALSHI_API_KEY",     None),
            private_key = getattr(Config, "KALSHI_PRIVATE_KEY", None),
            dry_run     = getattr(Config, "KALSHI_DRY_RUN",     True),
        )
        balance = executor.get_balance()
        if not balance:
            return jsonify({"success": False, "error": "Balance unavailable (no credentials?)"}), 503
        return jsonify({"success": True, "data": balance})
    except Exception as e:
        logger.error(f"Kalshi balance fetch failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@kalshi_bp.route("/status", methods=["GET"])
def get_status():
    """Health check for Kalshi executor and predictor."""
    try:
        from ..services.kalshi_executor import KalshiExecutor
        executor = KalshiExecutor(
            api_key     = getattr(Config, "KALSHI_API_KEY",     None),
            private_key = getattr(Config, "KALSHI_PRIVATE_KEY", None),
            dry_run     = getattr(Config, "KALSHI_DRY_RUN",     True),
        )
        exec_health = executor.health_check()
        pred        = _get_predictor()
        pred_health = {
            "is_trained":      pred.is_trained,
            "training_stats":  pred.training_stats,
        }

        return jsonify({
            "success": True,
            "data": {
                "executor":  exec_health,
                "predictor": pred_health,
                "credentials_present": bool(
                    getattr(Config, "KALSHI_API_KEY",     None) and
                    getattr(Config, "KALSHI_PRIVATE_KEY", None)
                ),
            }
        })

    except Exception as e:
        logger.error(f"Kalshi status check failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
