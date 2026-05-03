"""
Kalshi Executor — Place trades on Kalshi's Trading API v2

Order concepts:
  - Prices are in CENTS (1–99), not probabilities
  - Quantity is in CONTRACTS (each contract pays $1 if it resolves YES, $0 if NO)
  - To buy YES at 45¢ for $100 risk → count = floor(100 / 0.45) ≈ 222 contracts
  - Max loss = count * yes_price_cents / 100 (buying YES)
  - Max gain = count * (1 - yes_price_cents / 100) (buying YES)

Authentication:
  RSA-PKCS1v15/SHA-256 signed headers (same as KalshiDataFetcher).
  Requires KALSHI_API_KEY and KALSHI_PRIVATE_KEY in .env.

dry_run=True (default) simulates orders without hitting the exchange.
"""

import base64
import logging
import time
import math
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..config import Config

logger = logging.getLogger("mirofish.kalshi_executor")

_PROD_BASE = "https://trading-api.kalshi.com/trade-api/v2"
_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _base_url() -> str:
    return _DEMO_BASE if getattr(Config, "KALSHI_DEMO", False) else _PROD_BASE


def _sign(api_key_id: str, private_key_pem: str, method: str, path: str) -> Dict[str, str]:
    """Build RSA-PKCS1v15/SHA-256 auth headers."""
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

        pem = private_key_pem.replace("\\n", "\n")
        private_key = serialization.load_pem_private_key(pem.encode(), password=None)

        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + f"/trade-api/v2{path}").encode()
        sig = private_key.sign(msg, asym_padding.PKCS1v15(), hashes.SHA256())

        return {
            "KALSHI-ACCESS-KEY": api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "Content-Type": "application/json",
        }
    except Exception as e:
        logger.error(f"Kalshi RSA signing failed: {e}")
        return {"Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class KalshiExecutor:
    """
    Execute trades on Kalshi's Trading API v2.

    Args:
        api_key:    Kalshi API key ID (from dashboard)
        private_key: RSA private key PEM string
        dry_run:    Simulate orders without submitting (default True)
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 private_key: Optional[str] = None,
                 dry_run: bool = True):
        self.api_key     = api_key     or getattr(Config, "KALSHI_API_KEY",     None)
        self.private_key = private_key or getattr(Config, "KALSHI_PRIVATE_KEY", None)
        self.dry_run     = dry_run
        self.connected   = False
        self._orders: Dict[str, Dict] = {}  # order_id → order info (dry-run tracking)

        self._check_connection()

    def _check_connection(self):
        """Verify credentials by hitting the exchange status endpoint."""
        if not self.api_key or not self.private_key:
            logger.warning("Kalshi credentials not set — executor running in credential-less mode")
            return
        try:
            self._get("/exchange/status")
            self.connected = True
            mode = "DEMO" if getattr(Config, "KALSHI_DEMO", False) else "PRODUCTION"
            logger.info(f"KalshiExecutor connected ({mode}, dry_run={self.dry_run})")
        except Exception as e:
            logger.warning(f"Kalshi connection check failed: {e}")

    # =========================================================================
    # Internal HTTP
    # =========================================================================

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        if self.api_key and self.private_key:
            return _sign(self.api_key, self.private_key, method, path)
        return {"Content-Type": "application/json"}

    def _get(self, path: str, params: Dict = None) -> Any:
        url = _base_url() + path
        resp = requests.get(url, params=params or {}, headers=self._headers("GET", path), timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: Dict) -> Any:
        url = _base_url() + path
        resp = requests.post(url, json=body, headers=self._headers("POST", path), timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> Any:
        url = _base_url() + path
        resp = requests.delete(url, headers=self._headers("DELETE", path), timeout=15)
        resp.raise_for_status()
        return resp.json()

    # =========================================================================
    # Order placement
    # =========================================================================

    def place_order(self,
                    ticker: str,
                    side: str,
                    amount: float,
                    price: float,
                    kelly_fraction: float = 0.25) -> Optional[Dict[str, Any]]:
        """
        Place a limit order on Kalshi.

        Args:
            ticker:          Market ticker, e.g. "BTCUSD-24DEC31"
            side:            "BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO"
            amount:          USD dollar amount to risk
            price:           Limit price as probability [0, 1]
            kelly_fraction:  Kelly sizing multiplier (default quarter Kelly)

        Returns:
            Order confirmation dict or None on failure.
        """
        # Apply Kelly sizing
        sized_amount = amount * kelly_fraction
        if sized_amount < 1.0:
            logger.warning(f"Order size too small after Kelly ({sized_amount:.2f}) — skipping")
            return None

        # Parse side
        side_upper = side.upper()
        if side_upper not in ("BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO"):
            raise ValueError(f"Invalid side: {side}. Must be BUY_YES/BUY_NO/SELL_YES/SELL_NO")

        action     = "buy"  if "BUY"  in side_upper else "sell"
        yes_or_no  = "yes" if "YES"  in side_upper else "no"

        # Convert price probability → cents
        price_cents = max(1, min(99, round(price * 100)))

        # Calculate contract count
        # cost per contract when buying YES at price_cents: price_cents / 100 dollars
        cost_per_contract = price_cents / 100 if action == "buy" else (100 - price_cents) / 100
        if cost_per_contract <= 0:
            return None
        count = max(1, math.floor(sized_amount / cost_per_contract))

        order_body = {
            "ticker":    ticker,
            "action":    action,
            "side":      yes_or_no,
            "type":      "limit",
            "count":     count,
            "yes_price": price_cents,
        }

        if self.dry_run:
            order_id = f"dry_{ticker}_{int(time.time())}"
            order = {
                "order_id":       order_id,
                "status":         "dry_run",
                "ticker":         ticker,
                "side":           side,
                "amount_usd":     sized_amount,
                "price":          price,
                "price_cents":    price_cents,
                "count":          count,
                "kelly_fraction": kelly_fraction,
                "timestamp":      datetime.utcnow().isoformat(),
                "dry_run":        True,
            }
            self._orders[order_id] = order
            logger.info(f"[DRY RUN] Kalshi order: {ticker} {side} {count} contracts @ {price_cents}¢")
            return order

        # Live order
        if not self.api_key or not self.private_key:
            raise ValueError("KALSHI_API_KEY and KALSHI_PRIVATE_KEY are required to place live orders")

        try:
            resp = self._post("/portfolio/orders", order_body)
            raw_order = resp.get("order", resp)
            order = {
                "order_id":       raw_order.get("order_id", ""),
                "status":         raw_order.get("status", "resting"),
                "ticker":         ticker,
                "side":           side,
                "amount_usd":     sized_amount,
                "price":          price,
                "price_cents":    price_cents,
                "count":          count,
                "kelly_fraction": kelly_fraction,
                "timestamp":      datetime.utcnow().isoformat(),
                "dry_run":        False,
                "raw":            raw_order,
            }
            self._orders[order["order_id"]] = order
            logger.info(f"Kalshi order placed: {order['order_id']} {ticker} {side} {count}@{price_cents}¢")
            return order
        except Exception as e:
            logger.error(f"Kalshi order placement failed: {e}")
            return None

    # =========================================================================
    # Portfolio queries
    # =========================================================================

    def get_active_orders(self) -> List[Dict]:
        """Return open/resting orders from Kalshi (or dry-run cache)."""
        if self.dry_run:
            return [o for o in self._orders.values() if o.get("status") == "dry_run"]

        if not self.api_key:
            return []
        try:
            data = self._get("/portfolio/orders", {"status": "resting"})
            return data.get("orders", [])
        except Exception as e:
            logger.error(f"Failed to fetch Kalshi orders: {e}")
            return []

    def get_positions(self, ticker: str = None) -> List[Dict]:
        """Return open positions, optionally filtered by ticker."""
        if self.dry_run:
            return []   # dry-run has no settled positions

        if not self.api_key:
            return []
        try:
            params = {}
            if ticker:
                params["ticker"] = ticker
            data = self._get("/portfolio/positions", params)
            positions = data.get("market_positions", [])
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch Kalshi positions: {e}")
            return []

    def get_balance(self) -> Optional[Dict]:
        """Return account cash balance."""
        if not self.api_key:
            return None
        try:
            data = self._get("/portfolio/balance")
            balance_cents = data.get("balance", 0)
            return {
                "balance_usd":   round(balance_cents / 100, 2),
                "balance_cents": balance_cents,
            }
        except Exception as e:
            logger.error(f"Failed to fetch Kalshi balance: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order by ID."""
        if self.dry_run:
            if order_id in self._orders:
                self._orders[order_id]["status"] = "cancelled"
                return True
            return False

        if not self.api_key:
            return False
        try:
            self._delete(f"/portfolio/orders/{order_id}")
            if order_id in self._orders:
                self._orders[order_id]["status"] = "cancelled"
            return True
        except Exception as e:
            logger.error(f"Failed to cancel Kalshi order {order_id}: {e}")
            return False

    # =========================================================================
    # Health check
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """Return a summary of executor state."""
        exchange_ok = False
        if self.api_key and self.private_key:
            try:
                status = self._get("/exchange/status")
                exchange_ok = status.get("trading_active", False)
            except Exception:
                pass

        mode = "DEMO" if getattr(Config, "KALSHI_DEMO", False) else "PRODUCTION"
        return {
            "connected":       self.connected,
            "exchange_active": exchange_ok,
            "dry_run":         self.dry_run,
            "mode":            mode,
            "has_api_key":     bool(self.api_key),
            "has_private_key": bool(self.private_key),
            "open_orders":     len([o for o in self._orders.values() if o.get("status") == "dry_run"]),
            "timestamp":       datetime.utcnow().isoformat(),
        }
