"""
Polymarket Data Fetcher

Connects to Polymarket's public Gamma API to fetch:
  1. Active markets with current probabilities
  2. Market details (question, outcomes, liquidity)
  3. Price history

Polymarket APIs used here:
  - Gamma API: gamma-api.polymarket.com  — market discovery & pricing (no key required)
  - CLOB API:  clob.polymarket.com        — order book / trading (handled by executor)

No API key is required for data fetching.
"""

import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..utils.cache import TTLCache, make_key

logger = logging.getLogger("mirofish.polymarket")

_GAMMA_API = "https://gamma-api.polymarket.com"
_TIMEOUT = 15

# Module-level cache shared across all PolymarketDataFetcher instances.
# TTL design:
#   market list (500 items) — 5 min: new markets open/close slowly
#   individual market       — 3 min: prices fluctuate, but small window is fine
_cache = TTLCache()

_TTL_MARKET_LIST = 300   # 5 minutes
_TTL_MARKET      = 180   # 3 minutes


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------

def _get(url: str, params: Dict = None, ttl: float = 0) -> Any:
    """
    GET with timeout; raises on HTTP error.
    If ttl > 0, the response is cached for `ttl` seconds.
    """
    if ttl > 0:
        key = make_key(url, params or {})
        cached = _cache.get(key)
        if cached is not None:
            return cached

    resp = requests.get(url, params=params or {}, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    if ttl > 0:
        _cache.set(key, data, ttl)

    return data


def _extract_list(data: Any) -> List[Dict]:
    """
    Normalize a Gamma API response to a plain list of market dicts.

    The Gamma API returns a bare JSON array — NOT {"data": [...]}.
    Handle both just in case future versions change the envelope.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("data", data.get("markets", []))
    return []


# ---------------------------------------------------------------------------
# Market normalization
# ---------------------------------------------------------------------------

def _parse_market(raw: Dict) -> Dict:
    """
    Normalize a raw Gamma API market object into a consistent shape.

    Key transformations applied here:
      - outcomes      : JSON string  → Python list of names  e.g. ['Yes', 'No']
      - outcomePrices : JSON string  → Python list of floats e.g. [0.65, 0.35]
      - yes_price / no_price : extracted from the above pairs
      - volume / liquidity   : string → float  (API returns numeric strings)
      - unique_traders       : None-safe int
      - days_to_close        : computed from endDate
    """
    # ---- outcome names -------------------------------------------------------
    outcomes_raw = raw.get("outcomes", "[]") or "[]"
    if isinstance(outcomes_raw, str):
        try:
            outcomes: List = json.loads(outcomes_raw)
        except Exception:
            outcomes = []
    else:
        outcomes = list(outcomes_raw)

    # ---- outcome prices -------------------------------------------------------
    prices_raw = raw.get("outcomePrices", "[]") or "[]"
    if isinstance(prices_raw, str):
        try:
            parsed_prices = json.loads(prices_raw)
        except Exception:
            parsed_prices = []
    else:
        parsed_prices = list(prices_raw)

    prices: List[float] = []
    for p in parsed_prices:
        try:
            prices.append(float(p))
        except (ValueError, TypeError):
            prices.append(0.0)

    # ---- match YES / NO to their prices --------------------------------------
    yes_price = 0.5
    no_price = 0.5
    for i, name in enumerate(outcomes):
        label = str(name).strip().lower()
        if label in ("yes", "true", "1") and i < len(prices):
            yes_price = prices[i]
        elif label in ("no", "false", "0") and i < len(prices):
            no_price = prices[i]

    # Clamp to valid probability range
    yes_price = max(0.01, min(0.99, yes_price)) if yes_price else 0.5
    no_price  = max(0.01, min(0.99, no_price))  if no_price  else 0.5

    # ---- numeric volume / liquidity ------------------------------------------
    # volumeNum / liquidityNum are already floats; fall back to string volume
    def _flt(v, default: float = 0.0) -> float:
        try:
            return float(v or 0)
        except (ValueError, TypeError):
            return default

    volume    = _flt(raw.get("volumeNum",    raw.get("volume",    0)))
    liquidity = _flt(raw.get("liquidityNum", raw.get("liquidity", 0)))

    # ---- unique traders (often None in the API) ------------------------------
    ut = raw.get("uniqueTraderCount", 0)
    try:
        unique_traders = int(ut or 0)
    except (ValueError, TypeError):
        unique_traders = 0

    # ---- days to close -------------------------------------------------------
    days_to_close = 30
    end_str = raw.get("endDateIso") or raw.get("endDate") or ""
    if end_str:
        try:
            clean = end_str.replace("Z", "").split(".")[0]
            end_dt = datetime.fromisoformat(clean)
            days_to_close = max(1, (end_dt - datetime.utcnow()).days)
        except Exception:
            pass

    # ---- CLOB token IDs (for order execution) --------------------------------
    clob_raw = raw.get("clobTokenIds", "[]") or "[]"
    if isinstance(clob_raw, str):
        try:
            clob_token_ids: List[str] = json.loads(clob_raw)
        except Exception:
            clob_token_ids = []
    else:
        clob_token_ids = list(clob_raw)

    yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
    no_token_id  = clob_token_ids[1] if len(clob_token_ids) > 1 else None

    # ---- category from events tag if top-level field missing -----------------
    category = raw.get("category", "")
    if not category:
        events = raw.get("events") or []
        if events and isinstance(events, list):
            tags = events[0].get("tags", []) if isinstance(events[0], dict) else []
            category = tags[0].get("label", "unknown") if tags else "unknown"
        else:
            category = "unknown"

    return {
        # Pass through all original fields so callers can access anything
        **raw,
        # Normalized / computed fields (override originals where needed)
        "id":             raw.get("id", ""),
        "question":       raw.get("question", ""),
        "category":       category,
        "outcomes":       outcomes,
        "yes_price":      yes_price,
        "no_price":       no_price,
        "volume":         volume,
        "liquidity":      liquidity,
        "unique_traders": unique_traders,
        "days_to_close":  days_to_close,
        "expires_at":     end_str,
        "yes_token_id":   yes_token_id,
        "no_token_id":    no_token_id,
    }


# ---------------------------------------------------------------------------
# Data Fetcher
# ---------------------------------------------------------------------------

class PolymarketDataFetcher:
    """
    Fetches market data from Polymarket's Gamma REST API (completely free, no key required).
    """

    def __init__(self):
        self.base_url = _GAMMA_API

    # =========================================================================
    # Markets
    # =========================================================================

    def get_active_markets(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Fetch list of active, non-closed markets (cached 5 min)."""
        try:
            params = {
                "limit":  limit,
                "offset": offset,
                "active": "true",
                "closed": "false",
            }
            data = _get(f"{self.base_url}/markets", params, ttl=_TTL_MARKET_LIST)
            return [_parse_market(m) for m in _extract_list(data)]
        except Exception as e:
            logger.warning(f"Failed to fetch active markets: {e}")
            return []

    def get_market(self, market_id: str) -> Optional[Dict]:
        """Fetch and normalize details for a single market (cached 3 min)."""
        try:
            data = _get(f"{self.base_url}/markets/{market_id}", ttl=_TTL_MARKET)
            if isinstance(data, dict):
                return _parse_market(data)
            lst = _extract_list(data)
            return _parse_market(lst[0]) if lst else None
        except Exception as e:
            logger.warning(f"Failed to fetch market {market_id}: {e}")
            return None

    def get_market_metadata(self, market_id: str) -> Optional[Dict]:
        """Alias for get_market — used by the API blueprint."""
        return self.get_market(market_id)

    def search_markets(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for markets matching a keyword (cached 5 min)."""
        try:
            params = {"keyword": query, "limit": limit}
            data = _get(f"{self.base_url}/markets", params, ttl=_TTL_MARKET_LIST)
            return [_parse_market(m) for m in _extract_list(data)]
        except Exception as e:
            logger.warning(f"Market search failed for '{query}': {e}")
            return []

    def get_markets_by_category(self, category: str, limit: int = 50) -> List[Dict]:
        """
        Fetch markets in a category (cached 5 min).
        Uses Gamma API `tag` parameter — valid values: politics, sports, crypto,
        economics, world, science, entertainment, etc.
        """
        try:
            params = {"tag": category, "limit": limit, "active": "true"}
            data = _get(f"{self.base_url}/markets", params, ttl=_TTL_MARKET_LIST)
            return [_parse_market(m) for m in _extract_list(data)]
        except Exception as e:
            logger.warning(f"Failed to fetch markets for category '{category}': {e}")
            return []

    # =========================================================================
    # Pricing
    # =========================================================================

    def get_current_price(self, market_id: str) -> Optional[Dict]:
        """
        Return current YES/NO prices for a market.

        Returns: {market_id, yes_price, no_price, mid_price, spread, timestamp}
        """
        market = self.get_market(market_id)
        if not market:
            return None
        yes_price = market["yes_price"]
        no_price  = market["no_price"]
        return {
            "market_id":  market_id,
            "yes_price":  yes_price,
            "no_price":   no_price,
            "mid_price":  round((yes_price + no_price) / 2, 4),
            "spread":     round(abs(yes_price - no_price), 4),
            "timestamp":  market.get("expires_at", datetime.utcnow().isoformat()),
        }

    def get_prices(self, market_id: str, hours: int = 24) -> List[Dict]:
        """
        Fetch price history for a market (best effort — returns [] if unavailable).
        Uses Gamma API prices-history endpoint.
        """
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            params = {
                "market":    market_id,
                "startTs":   int(since.timestamp()),
                "interval":  "1h",
                "fidelity":  24,
            }
            data = _get(f"{self.base_url}/prices-history", params)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Failed to fetch price history for {market_id}: {e}")
            return []

    # =========================================================================
    # Order book & recent orders
    # =========================================================================

    def get_order_book(self, market_id: str) -> Optional[Dict]:
        """
        CLOB order book requires a token_id (not a market_id) and is served by
        clob.polymarket.com — handled by the executor.  Returns None here.
        """
        return None

    def get_recent_orders(self, market_id: str, limit: int = 50) -> List[Dict]:
        """Fetch recent trades/fills for a market (best effort)."""
        try:
            params = {"market": market_id, "limit": limit}
            data = _get(f"{self.base_url}/trades", params)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("data", [])
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch recent orders for {market_id}: {e}")
            return []

    def get_market_volume(self, market_id: str, period: str = "24h") -> Optional[Dict]:
        """Return volume and liquidity metrics for a market."""
        market = self.get_market(market_id)
        if not market:
            return None
        return {
            "market_id":      market_id,
            "volume_24h":     market.get("volume24hr", market.get("volume", 0)),
            "volume_total":   market.get("volume", 0),
            "liquidity":      market.get("liquidity", 0),
            "unique_traders": market.get("unique_traders", 0),
            "liquidity_score": self._compute_liquidity_score(market),
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _compute_liquidity_score(self, market: Dict) -> float:
        """0-100 liquidity score derived from 24h volume."""
        volume = float(market.get("volume24hr") or market.get("volume") or 0)
        if volume < 1_000:
            return round((volume / 1_000) * 20, 2)
        elif volume < 5_000:
            return round(20 + (volume - 1_000) / 4_000 * 30, 2)
        elif volume < 20_000:
            return round(50 + (volume - 5_000) / 15_000 * 30, 2)
        else:
            return min(100.0, round(80 + (volume - 20_000) / 100_000 * 20, 2))

    def filter_high_liquidity_markets(self,
                                       min_volume: float = 5_000,
                                       min_traders: int = 0,
                                       fetch_limit: int = 200) -> List[Dict]:
        """
        Return markets with sufficient volume for clean execution.

        fetch_limit caps how many markets are fetched from the API (default 200).
        This keeps the scan fast and cheap; the 200 most-recently-updated markets
        cover the liquid tail well since Polymarket returns them sorted by recency.

        Note: uniqueTraderCount is often None in the Gamma API, so min_traders
        defaults to 0.  Pass min_traders > 0 only if you have reason to believe
        that field is populated for your target markets.
        """
        all_markets = self.get_active_markets(limit=fetch_limit)
        result = []
        for m in all_markets:
            vol = float(m.get("volume", 0) or 0)
            if vol < min_volume:
                continue
            if min_traders > 0:
                traders = int(m.get("unique_traders", 0) or 0)
                # If uniqueTraderCount is unknown (0), don't reject the market
                if traders > 0 and traders < min_traders:
                    continue
            result.append(m)
        return result

    def find_mispriced_markets(self, predictor) -> List[Dict]:
        """
        Scan high-liquidity markets and return those with the most edge
        according to a PolymarketPredictor instance.

        Sorting: largest absolute edge first.
        """
        markets = self.filter_high_liquidity_markets(min_volume=1_000, fetch_limit=200)
        predictions = []

        for market in markets:
            market_id = str(market.get("id", ""))
            if not market_id:
                continue
            try:
                pred = predictor.predict(
                    market_id=market_id,
                    question=market.get("question", ""),
                    yes_price=market.get("yes_price", 0.5),
                    no_price=market.get("no_price", 0.5),
                    volume=market.get("volume", 0),
                    category=market.get("category", "unknown"),
                    time_to_close_days=market.get("days_to_close", 30),
                )
                edge = abs(pred.get("yes_probability", 0.5) - market.get("yes_price", 0.5))
                predictions.append({
                    "market_id":         market_id,
                    "question":          market.get("question", ""),
                    "market_yes_price":  market.get("yes_price", 0.5),
                    "predicted_yes_prob": pred.get("yes_probability", 0.5),
                    "edge":              edge,
                    "edge_signal":       pred.get("edge_signal", "NEUTRAL"),
                    "kelly_fraction":    pred.get("kelly_fraction", 0),
                    "volume":            market.get("volume", 0),
                    "prediction":        pred,
                })
            except Exception as e:
                logger.warning(f"Prediction failed for {market_id}: {e}")

        predictions.sort(key=lambda x: x["edge"], reverse=True)
        return predictions
