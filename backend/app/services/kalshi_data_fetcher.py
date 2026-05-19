"""
Kalshi Data Fetcher

Connects to Kalshi's Trading API v2 to fetch live market data:
  1. Active markets with current bid/ask prices
  2. Single-market detail lookup
  3. Keyword search (client-side filter on fetched results)

Kalshi API:
  - Production : https://trading-api.kalshi.com/trade-api/v2
  - Demo       : https://demo-api.kalshi.co/trade-api/v2

Authentication:
  All Kalshi v2 endpoints require RSA-PKCS1v15/SHA-256 signed headers:
    KALSHI-ACCESS-KEY        — API key ID (from Kalshi dashboard)
    KALSHI-ACCESS-TIMESTAMP  — Unix ms timestamp
    KALSHI-ACCESS-SIGNATURE  — base64(RSA-SHA256(key, ts + METHOD + /trade-api/v2<path>))

  If credentials are absent the fetcher logs a warning and returns empty
  results so the rest of the app continues to work.

Price convention:
  Kalshi returns prices in CENTS (1–99).  This module divides by 100 so all
  downstream code sees probabilities in [0.01, 0.99].
"""

import base64
import logging
import time
import requests
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.cache import TTLCache, make_key

logger = logging.getLogger("mirofish.kalshi")

_PROD_BASE = "https://trading-api.kalshi.com/trade-api/v2"
_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
_TIMEOUT = 15

_cache = TTLCache()
_TTL_MARKET_LIST = 300  # 5 minutes
_TTL_MARKET = 180       # 3 minutes


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _base_url() -> str:
    return _DEMO_BASE if getattr(Config, "KALSHI_DEMO", False) else _PROD_BASE


def _sign_request(api_key_id: str, private_key_pem: str, method: str, path: str) -> Dict[str, str]:
    """
    Build RSA-PKCS1v15/SHA-256 auth headers.

    path must be the full URL path including the /trade-api/v2 prefix but
    WITHOUT query parameters, e.g. '/trade-api/v2/markets'.
    """
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

        # Handle \\n-escaped newlines from .env files
        pem = private_key_pem.replace("\\n", "\n")
        private_key = serialization.load_pem_private_key(pem.encode(), password=None)

        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode()
        sig = private_key.sign(
            msg,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "Content-Type": "application/json",
        }
    except Exception as e:
        logger.error(f"Kalshi RSA signing failed: {e}")
        return {"Content-Type": "application/json"}


def _has_credentials() -> bool:
    return bool(getattr(Config, "KALSHI_API_KEY", None) and
                getattr(Config, "KALSHI_PRIVATE_KEY", None))


def _auth_headers(method: str, path: str) -> Dict[str, str]:
    """Return signed headers if credentials exist, otherwise bare Content-Type."""
    if not _has_credentials():
        return {"Content-Type": "application/json"}
    return _sign_request(Config.KALSHI_API_KEY, Config.KALSHI_PRIVATE_KEY, method, path)


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------

def _get(path: str, params: Dict = None, ttl: float = 0) -> Any:
    """Signed GET with optional TTL caching."""
    url = _base_url() + path

    if ttl > 0:
        key = make_key(url, params or {})
        cached = _cache.get(key)
        if cached is not None:
            return cached

    # path for signing must include /trade-api/v2 prefix
    headers = _auth_headers("GET", f"/trade-api/v2{path}")
    resp = requests.get(url, params=params or {}, headers=headers, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    if ttl > 0:
        key = make_key(url, params or {})
        _cache.set(key, data, ttl)

    return data


# ---------------------------------------------------------------------------
# Category inference from event_ticker prefix
# ---------------------------------------------------------------------------

_TICKER_CATEGORY_MAP = [
    ("KXMLB",  "sports"),
    ("KXNBA",  "sports"),
    ("KXNFL",  "sports"),
    ("KXNHL",  "sports"),
    ("KXNCC",  "sports"),
    ("KXCFB",  "sports"),
    ("KXMLS",  "sports"),
    ("KXSOC",  "sports"),
    ("KXTEN",  "sports"),
    ("KXMMA",  "sports"),
    ("KXBOX",  "sports"),
    ("KXNAS",  "sports"),
    ("KXF1",   "sports"),
    ("KXPGA",  "sports"),
    ("KXGOLF", "sports"),
    ("KXEPL",  "sports"),
    ("INX",    "economics"),
    ("SP500",  "economics"),
    ("NASDAQ", "economics"),
    ("FOMC",   "economics"),
    ("CPI",    "economics"),
    ("NFP",    "economics"),
    ("GDP",    "economics"),
    ("FED",    "economics"),
    ("OIL",    "economics"),
    ("GOLD",   "economics"),
    ("BTC",    "crypto"),
    ("ETH",    "crypto"),
    ("CRYPTO", "crypto"),
    ("PRES",   "politics"),
    ("SEN",    "politics"),
    ("HOUSE",  "politics"),
    ("REPUB",  "politics"),
    ("DEMO",   "politics"),
    ("ELEC",   "politics"),
    ("TEMP",   "weather"),
    ("WIND",   "weather"),
    ("RAIN",   "weather"),
    ("HURR",   "weather"),
]


def _infer_category(raw: Dict) -> str:
    """Derive a human category from the API category field or event_ticker prefix."""
    cat = (raw.get("category") or "").strip().lower()
    if cat and cat != "unknown":
        return cat
    event_ticker = (raw.get("event_ticker") or raw.get("ticker") or "").upper()
    for prefix, category in _TICKER_CATEGORY_MAP:
        if event_ticker.startswith(prefix):
            return category
    return "other"


# ---------------------------------------------------------------------------
# Market normalization
# ---------------------------------------------------------------------------

def _parse_market(raw: Dict) -> Dict:
    """
    Normalize a raw Kalshi v2 market object to the shape used across MiroFish.

    Key transformations:
      - Cents (1-99) → probabilities (0.01-0.99) for yes/no prices
      - None prices (illiquid markets) → None, with has_price=False flag
      - close_time ISO string → days_to_close int
      - Category derived from event_ticker when API returns null
    """
    def _cents_to_prob(v) -> Optional[float]:
        """Convert a cents integer (1-99) to probability, or None if missing/zero."""
        if v is None:
            return None
        try:
            c = int(v)
            return round(c / 100, 4) if c > 0 else None
        except (ValueError, TypeError):
            return None

    def _flt(v, default: float = 0.0) -> float:
        try:
            return float(v or 0)
        except (ValueError, TypeError):
            return default

    yes_bid = _cents_to_prob(raw.get("yes_bid"))
    yes_ask = _cents_to_prob(raw.get("yes_ask"))
    no_bid  = _cents_to_prob(raw.get("no_bid"))
    no_ask  = _cents_to_prob(raw.get("no_ask"))

    # Compute mid-prices only when bid+ask both exist; otherwise use last price
    has_price = yes_bid is not None and yes_ask is not None
    if has_price:
        yes_price = round(max(0.01, min(0.99, (yes_bid + yes_ask) / 2)), 4)
        no_price  = round(max(0.01, min(0.99, (no_bid  + no_ask)  / 2 if (no_bid and no_ask) else 1 - yes_price)), 4)
    else:
        # Fall back to last_price_dollars if available
        last_str = raw.get("last_price_dollars") or raw.get("previous_price_dollars")
        if last_str:
            try:
                lp = float(last_str)
                if 0 < lp < 1:
                    yes_price = round(lp, 4)
                    no_price  = round(1 - lp, 4)
                    has_price = True
                else:
                    yes_price = None
                    no_price  = None
            except (ValueError, TypeError):
                yes_price = None
                no_price  = None
        else:
            yes_price = None
            no_price  = None

    # days to close
    days_to_close = 30
    close_str = raw.get("close_time", "") or ""
    if close_str:
        try:
            close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            delta_seconds = (close_dt - datetime.now(timezone.utc)).total_seconds()
            days_to_close = max(0.01, delta_seconds / 86400.0)
        except Exception:
            pass

    volume     = _flt(raw.get("volume", 0))
    volume_24h = _flt(raw.get("volume_24h", 0))
    liquidity  = _flt(raw.get("liquidity", 0))

    ticker   = raw.get("ticker", "")
    title    = raw.get("title", "")
    category = _infer_category(raw)

    return {
        **raw,
        "id":            ticker,
        "ticker":        ticker,
        "question":      title,
        "title":         title,
        "category":      category,
        "yes_price":     yes_price,
        "no_price":      no_price,
        "yes_bid":       yes_bid,
        "yes_ask":       yes_ask,
        "no_bid":        no_bid,
        "no_ask":        no_ask,
        "has_price":     has_price,
        "volume":        volume,
        "volume_24h":    volume_24h,
        "liquidity":     liquidity,
        "days_to_close": days_to_close,
        "expires_at":    close_str,
        "status":        raw.get("status", "unknown"),
    }


# ---------------------------------------------------------------------------
# Data Fetcher
# ---------------------------------------------------------------------------

class KalshiDataFetcher:
    """
    Fetches live market data from Kalshi's Trading API v2.

    Requires KALSHI_API_KEY and KALSHI_PRIVATE_KEY in .env.
    Degrades gracefully to empty results if credentials are absent.
    """

    # =========================================================================
    # Markets
    # =========================================================================

    def get_active_markets(self, limit: int = 100, cursor: str = "") -> List[Dict]:
        """Fetch active (open) markets, normalized.  Cached 5 min."""
        if not _has_credentials():
            logger.warning("KALSHI_API_KEY / KALSHI_PRIVATE_KEY not set — cannot fetch markets")
            return []
        try:
            params: Dict = {"status": "open", "limit": min(limit, 1000)}
            if cursor:
                params["cursor"] = cursor
            data = _get("/markets", params, ttl=_TTL_MARKET_LIST)
            raw_markets = data.get("markets", [])
            return [_parse_market(m) for m in raw_markets]
        except Exception as e:
            logger.warning(f"Kalshi active markets fetch failed: {e}")
            return []

    def get_market(self, ticker: str) -> Optional[Dict]:
        """Fetch and normalize a single market by ticker.  Cached 3 min."""
        if not _has_credentials():
            logger.warning("KALSHI_API_KEY / KALSHI_PRIVATE_KEY not set")
            return None
        try:
            data = _get(f"/markets/{ticker}", ttl=_TTL_MARKET)
            raw = data.get("market", data)
            return _parse_market(raw) if isinstance(raw, dict) else None
        except Exception as e:
            logger.warning(f"Kalshi market fetch failed for {ticker}: {e}")
            return None

    def get_market_metadata(self, ticker: str) -> Optional[Dict]:
        """Alias for get_market — used by the API blueprint."""
        return self.get_market(ticker)

    def search_markets(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search markets by keyword (client-side filter on fetched results).
        Kalshi v2 does not expose a dedicated search endpoint.
        """
        q = query.lower()
        markets = self.get_active_markets(limit=200)
        results = [
            m for m in markets
            if q in m.get("question", "").lower() or q in m.get("ticker", "").lower()
        ]
        return results[:limit]

    def get_markets_by_category(self, category: str, limit: int = 50) -> List[Dict]:
        """Fetch markets in a given category (e.g. 'economics', 'politics')."""
        if not _has_credentials():
            return []
        try:
            params = {"status": "open", "category": category.lower(), "limit": min(limit, 1000)}
            data = _get("/markets", params, ttl=_TTL_MARKET_LIST)
            return [_parse_market(m) for m in data.get("markets", [])]
        except Exception as e:
            logger.warning(f"Kalshi category fetch failed for '{category}': {e}")
            return []

    # =========================================================================
    # Pricing
    # =========================================================================

    def get_current_price(self, ticker: str) -> Optional[Dict]:
        """Return current YES/NO mid-prices for a market."""
        market = self.get_market(ticker)
        if not market:
            return None
        yes_price = market.get("yes_price")
        no_price  = market.get("no_price")
        if yes_price is None and no_price is None:
            return {
                "ticker":    ticker,
                "market_id": ticker,
                "yes_price": None,
                "no_price":  None,
                "mid_price": None,
                "spread":    None,
                "has_price": False,
                "timestamp": market.get("expires_at", datetime.utcnow().isoformat()),
            }
        yes_bid = market.get("yes_bid") or 0
        yes_ask = market.get("yes_ask") or 0
        return {
            "ticker":    ticker,
            "market_id": ticker,
            "yes_price": yes_price,
            "no_price":  no_price,
            "mid_price": round((yes_price + (no_price or 1 - yes_price)) / 2, 4),
            "spread":    round(abs(yes_ask - yes_bid), 4),
            "has_price": True,
            "timestamp": market.get("expires_at", datetime.utcnow().isoformat()),
        }

    # =========================================================================
    # Opportunity scanning
    # =========================================================================

    def filter_high_liquidity_markets(self,
                                       min_volume: float = 1_000,
                                       fetch_limit: int = 200) -> List[Dict]:
        """Return markets with volume above threshold."""
        all_markets = self.get_active_markets(limit=fetch_limit)
        return [m for m in all_markets if float(m.get("volume", 0) or 0) >= min_volume]

    def find_mispriced_markets(self, predictor) -> List[Dict]:
        """
        Scan high-liquidity markets and return those with the largest edge
        according to a KalshiPredictor-compatible predictor instance.
        """
        markets = self.filter_high_liquidity_markets(min_volume=1_000, fetch_limit=200)
        results = []
        for m in markets:
            ticker = m.get("ticker", "")
            if not ticker:
                continue
            try:
                pred = predictor.predict(
                    market_question=m.get("question", ""),
                    community_prob=m.get("yes_price", 0.5),
                    category=m.get("category", "unknown"),
                    time_to_close_days=m.get("days_to_close", 30),
                )
                edge = abs(pred.get("yes_probability", 0.5) - m.get("yes_price", 0.5))
                results.append({
                    "ticker":             ticker,
                    "question":           m.get("question", ""),
                    "market_yes_price":   m.get("yes_price", 0.5),
                    "predicted_yes_prob": pred.get("yes_probability", 0.5),
                    "edge":               round(edge, 4),
                    "edge_signal":        pred.get("edge_signal", "NEUTRAL"),
                    "kelly_fraction":     pred.get("kelly_fraction", 0),
                    "volume":             m.get("volume", 0),
                    "prediction":         pred,
                })
            except Exception as e:
                logger.warning(f"Prediction failed for {ticker}: {e}")
        results.sort(key=lambda x: x["edge"], reverse=True)
        return results

    # =========================================================================
    # Helpers
    # =========================================================================

    def _compute_liquidity_score(self, market: Dict) -> float:
        """0–100 score derived from 24h volume (mirrors Polymarket convention)."""
        volume = float(market.get("volume_24h") or market.get("volume") or 0)
        if volume < 1_000:
            return round((volume / 1_000) * 20, 2)
        elif volume < 5_000:
            return round(20 + (volume - 1_000) / 4_000 * 30, 2)
        elif volume < 20_000:
            return round(50 + (volume - 5_000) / 15_000 * 30, 2)
        return min(100.0, round(80 + (volume - 20_000) / 100_000 * 20, 2))
