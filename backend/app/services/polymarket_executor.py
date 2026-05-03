"""
Polymarket Executor — Place trades on Polymarket CLOB

Uses py-clob-client to:
  1. Connect to Polygon blockchain
  2. Sign orders with private key
  3. Place buy/sell orders on the CLOB
  4. Track order status and fills
  5. Manage positions and P&L

Key concepts:
  - USDC on Polygon for payments (6 decimals)
  - Orders signed with Polygon wallet key
  - Outcomes are binary: YES (token 0) / NO (token 1)
  - Price is always in [0, 1] range (representing probability)

This layer trusts the predictor output and executes based on:
  - edge_signal: "BUY YES" / "BUY NO" / "NEUTRAL"
  - kelly_fraction: position sizing
  - suggested_bet_size: risk management
"""

import logging
import os
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime, timedelta

logger = logging.getLogger("mirofish.polymarket_executor")

# Optional: import py-clob-client when available
try:
    from py_clob_client.client import ClobClient
    # OrderArgs / BUY / SELL live in different locations depending on package version
    try:
        from py_clob_client.clob_types import OrderArgs, OrderType
    except ImportError:
        from py_clob_client.order_builder.types import OrderArgs, OrderType  # type: ignore
    try:
        from py_clob_client.order_builder.constants import BUY, SELL
    except ImportError:
        BUY, SELL = "BUY", "SELL"  # type: ignore
    CLOB_AVAILABLE = True
except ImportError:
    CLOB_AVAILABLE = False
    logger.warning("py-clob-client not installed. Install with: pip install py-clob-client")


class PolymarketExecutor:
    """
    Execute trades on Polymarket CLOB.

    Requires:
      - POLYMARKET_PRIVATE_KEY: Polygon wallet private key (hex string, no 0x prefix)
      - POLYMARKET_API_KEY: API key for order submission (optional, for REST mode)

    Two modes:
      1. CLOB Mode (recommended): Direct blockchain ordering via py-clob-client
         - Sign orders with private key
         - Submit to CLOB
         - Fully self-custodied

      2. REST Mode (fallback): Use Polymarket REST API for order placement
         - Requires API key
         - Less control but simpler setup
    """

    def __init__(self,
                 private_key: Optional[str] = None,
                 api_key: Optional[str] = None,
                 mode: str = "clob",
                 dry_run: bool = True):
        """
        Initialize executor.

        Args:
            private_key: Polygon wallet private key (hex, no 0x)
            api_key: Polymarket API key (for REST mode)
            mode: "clob" or "rest"
            dry_run: If True, simulate trades without submitting to blockchain
        """
        self.private_key = private_key or os.environ.get("POLYMARKET_PRIVATE_KEY")
        self.api_key = api_key or os.environ.get("POLYMARKET_API_KEY")
        self.mode = mode
        self.dry_run = dry_run
        self.client = None
        self.connected = False
        self.orders = {}  # Track placed orders: {order_id: {status, market_id, ...}}

        if self.mode == "clob":
            self._init_clob_client()
        logger.info(f"PolymarketExecutor initialized: mode={mode}, dry_run={dry_run}")

    def _init_clob_client(self):
        """Initialize py-clob-client for blockchain trading."""
        if not CLOB_AVAILABLE:
            logger.warning("py-clob-client not available. Cannot use CLOB mode.")
            self.mode = "rest"
            return

        if not self.private_key:
            raise ValueError("POLYMARKET_PRIVATE_KEY required for CLOB mode")

        try:
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(10)
            try:
                self.client = ClobClient(
                    host="https://clob.polymarket.com",
                    chain_id=137,  # Polygon mainnet
                    key=self.private_key,
                )
            finally:
                socket.setdefaulttimeout(old_timeout)
            self.connected = True
            logger.info("py-clob-client connected to Polymarket CLOB")
        except Exception as e:
            logger.error(f"Failed to initialize CLOB client: {e}")
            self.mode = "rest"

    # =========================================================================
    # Order Placement
    # =========================================================================

    def place_order(self,
                    market_id: str,
                    side: str,
                    amount: float,
                    price: float,
                    kelly_fraction: float = 0.25) -> Optional[Dict[str, Any]]:
        """
        Place an order on Polymarket.

        Args:
            market_id: Polymarket market ID
            side: "BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO"
            amount: Amount in USDC (e.g., 100)
            price: Limit price [0, 1] representing probability
            kelly_fraction: Kelly sizing (0.25 = quarter Kelly)

        Returns:
            Order dict: {order_id, status, market_id, side, amount, price, timestamp}
            or None if dry_run or error
        """
        if not amount or amount <= 0:
            logger.warning(f"Invalid amount: {amount}")
            return None

        # Apply Kelly sizing to position
        sized_amount = amount * kelly_fraction

        order = {
            "market_id": market_id,
            "side": side,
            "amount": sized_amount,
            "price": price,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending",
        }

        if self.dry_run:
            logger.info(f"[DRY RUN] Would place order: {side} {sized_amount} USDC @ {price}")
            order["status"] = "dry_run_accepted"
            return order

        try:
            if self.mode == "clob" and self.connected:
                return self._place_clob_order(market_id, side, sized_amount, price)
            elif self.mode == "rest" and self.api_key:
                return self._place_rest_order(market_id, side, sized_amount, price)
            else:
                logger.error(f"Cannot place order: mode={self.mode}, connected={self.connected}")
                return None
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            order["status"] = "failed"
            order["error"] = str(e)
            return order

    def _place_clob_order(self, market_id: str, side: str, amount: float, price: float) -> Dict:
        """
        Place order via py-clob-client (blockchain).

        Polymarket CLOB orders require an ERC-1155 token_id (one per outcome),
        NOT the market condition_id.  We look up the correct token by calling
        the CLOB /markets endpoint.
        """
        try:
            import requests as _req

            # Resolve the YES/NO token_id for this market
            clob_market = _req.get(
                f"https://clob.polymarket.com/markets/{market_id}",
                timeout=15,
            ).json()
            tokens = clob_market.get("tokens", [])

            is_yes = "YES" in side
            token_id = None
            for token in tokens:
                outcome_name = token.get("outcome", "").upper()
                if is_yes and outcome_name == "YES":
                    token_id = token.get("token_id")
                    break
                elif not is_yes and outcome_name == "NO":
                    token_id = token.get("token_id")
                    break

            if not token_id:
                raise ValueError(f"Could not resolve token_id for side={side}, market={market_id}")

            order_side = BUY if "BUY" in side else SELL
            order_args = OrderArgs(
                token_id=token_id,
                price=float(price),
                size=float(amount),
                side=order_side,
            )

            # create_and_post_order signs and submits in one call
            resp = self.client.create_and_post_order(order_args)
            order_id = resp.get("orderID") or resp.get("id") or "unknown"

            logger.info(f"CLOB order placed: id={order_id}, side={side}, amount={amount}")

            return {
                "order_id":  order_id,
                "status":    "submitted",
                "market_id": market_id,
                "side":      side,
                "amount":    amount,
                "price":     price,
                "timestamp": datetime.utcnow().isoformat(),
                "token_id":  token_id,
            }
        except Exception as e:
            logger.error(f"CLOB order failed: {e}")
            raise

    def _place_rest_order(self, market_id: str, side: str, amount: float, price: float) -> Dict:
        """Place order via REST API (requires API key)."""
        import requests

        try:
            payload = {
                "market_id": market_id,
                "side": side,
                "amount": amount,
                "price": price,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}

            resp = requests.post(
                "https://api.polymarket.com/orders",
                json=payload,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()

            logger.info(f"REST order placed: id={result.get('id')}, side={side}, amount={amount}")

            return {
                "order_id": result.get("id"),
                "status": "submitted",
                "market_id": market_id,
                "side": side,
                "amount": amount,
                "price": price,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"REST order failed: {e}")
            return {
                "order_id": None,
                "status": "failed",
                "market_id": market_id,
                "side": side,
                "amount": amount,
                "price": price,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            }

    # =========================================================================
    # Order Management
    # =========================================================================

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Check status of a placed order."""
        try:
            if self.mode == "clob" and self.connected:
                # CLOB: query blockchain
                status = self.client.get_order_status(order_id)
                return {
                    "order_id": order_id,
                    "status": status,
                    "filled": True if status == "filled" else False,
                }
            elif self.mode == "rest" and self.api_key:
                # REST: query API
                import requests
                resp = requests.get(
                    f"https://api.polymarket.com/orders/{order_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=15,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning(f"Failed to get order status: {e}")
        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would cancel order: {order_id}")
                return True

            if self.mode == "clob" and self.connected:
                self.client.cancel_order(order_id)
                logger.info(f"CLOB order cancelled: {order_id}")
                return True
            elif self.mode == "rest" and self.api_key:
                import requests
                resp = requests.delete(
                    f"https://api.polymarket.com/orders/{order_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=15,
                )
                resp.raise_for_status()
                logger.info(f"REST order cancelled: {order_id}")
                return True
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
        return False

    def get_active_orders(self) -> List[Dict]:
        """Get all active (unfilled) orders."""
        try:
            if self.mode == "clob" and self.connected:
                orders = self.client.get_orders()
                return orders
            elif self.mode == "rest" and self.api_key:
                import requests
                resp = requests.get(
                    "https://api.polymarket.com/orders",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    params={"status": "active"},
                    timeout=15,
                )
                resp.raise_for_status()
                return resp.json().get("data", [])
        except Exception as e:
            logger.warning(f"Failed to get active orders: {e}")
        return []

    # =========================================================================
    # Position Management
    # =========================================================================

    def get_balance(self) -> Optional[float]:
        """Get USDC balance (free + locked in orders)."""
        try:
            if self.mode == "clob" and self.connected:
                balance = self.client.get_balance()
                return float(balance) / 1_000_000  # Convert from 6 decimals to USDC
            # For REST, balance tracking requires additional state management
        except Exception as e:
            logger.warning(f"Failed to get balance: {e}")
        return None

    def get_positions(self, market_id: Optional[str] = None) -> List[Dict]:
        """
        Get current positions.

        If market_id: specific market position
        Otherwise: all open positions
        """
        try:
            if self.mode == "clob" and self.connected:
                positions = self.client.get_positions(market_id) if market_id else self.client.get_positions()
                return positions
        except Exception as e:
            logger.warning(f"Failed to get positions: {e}")
        return []

    def close_position(self, market_id: str, side: str) -> Optional[Dict]:
        """Close (flatten) a position by selling back shares."""
        logger.info(f"Closing position: {market_id} {side}")
        # This would create an offsetting order
        # E.g., if long YES, create a SELL_YES order for the position size
        # Implementation depends on position tracking
        return None

    # =========================================================================
    # Utilities
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """Check executor health status."""
        return {
            "mode": self.mode,
            "connected": self.connected,
            "dry_run": self.dry_run,
            "clob_available": CLOB_AVAILABLE,
            "has_private_key": bool(self.private_key),
            "has_api_key": bool(self.api_key),
            "timestamp": datetime.utcnow().isoformat(),
        }
