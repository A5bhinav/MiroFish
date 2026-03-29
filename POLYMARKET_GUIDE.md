# Polymarket Integration Guide

## Overview

MiroFish now has end-to-end Polymarket prediction + trading capability. The architecture mirrors the Kalshi integration but adds blockchain execution via Polygon.

**3 Layers:**
1. **Data Layer** (`polymarket_data_fetcher.py`) — Fetch markets, prices, liquidity from Polymarket's public APIs
2. **Prediction Layer** (`polymarket_predictor.py`) — Calibrate probabilities, compute edge, output trading signals
3. **Execution Layer** (`polymarket_executor.py`) — Place orders on-chain or via REST API

---

## Architecture

### Layer 1: Data Fetcher

**Source:** Polymarket REST API (completely free, no key required)

**Data Model:**
```python
PolymarketDataFetcher()
  .get_active_markets(limit=100)           # All live markets
  .search_markets("Fed rate cut")          # Keyword search
  .get_markets_by_category("economics")    # Category filter
  .get_market(market_id)                   # Single market detail
  .get_current_price(market_id)            # YES/NO prices
  .get_recent_orders(market_id)            # Trading activity
  .get_market_volume(market_id)            # Liquidity metrics
  .find_mispriced_markets(predictor)       # Auto-scan for edges
```

**Example Output:**
```json
{
  "id": "0x123...",
  "question": "Will the Fed cut rates in June 2026?",
  "category": "economics",
  "outcomes": [
    {"name": "Yes", "price": 0.35},
    {"name": "No", "price": 0.65}
  ],
  "volume": 52000,
  "unique_traders": 180,
  "expires_at": "2026-06-30T23:59:59Z",
  "created_at": "2026-03-15T10:00:00Z"
}
```

### Layer 2: Predictor

**Reuses KalshiPredictor's calibration engine**, adapted for Polymarket's CLOB pricing.

**Input:** Market question + current YES price
**Output:** Calibrated probability + edge signal + Kelly sizing

```python
predictor = PolymarketPredictor()

result = predictor.predict(
    market_id="0x123...",
    question="Will the Fed cut rates in June 2026?",
    yes_price=0.35,              # Market price
    no_price=0.65,
    volume=52000,                # 24h trading volume
    category="economics",
    time_to_close_days=95,
)

# Returns:
{
    "yes_probability": 0.42,     # Our calibrated estimate
    "no_probability": 0.58,
    "market_yes_price": 0.35,
    "edge": 0.07,                # We think YES is 7% underpriced
    "edge_signal": "BUY YES",
    "kelly_fraction": 0.25,      # Use quarter Kelly for risk management
    "suggested_order_size": 85,  # $85 USDC based on edge + liquidity
    "liquidity_score": 72,       # 0-100, higher = better
    "confidence": "medium",
    "factors": [...],
    "reasoning_summary": "Model assigns 42% YES vs market 35%. Buy YES..."
}
```

**Calibration Logic:**
- Base rate adjustments (Fed decisions, CPI thresholds)
- Economic context (FRED indicators)
- Temporal discounting (far-out markets are less predictable)
- Overconfidence shrinkage (compress extreme probabilities)
- Liquidity-adjusted confidence (thin markets = lower confidence)

### Layer 3: Executor

**Two Modes:**

**Mode A: CLOB (Direct Blockchain)**
- Uses `py-clob-client` to sign and place orders on Polygon
- Fully self-custodied, no intermediary
- Requires: `POLYMARKET_PRIVATE_KEY` (Polygon wallet key)
- Install: `pip install py-clob-client`

**Mode B: REST API**
- Uses Polymarket's REST API for order placement
- Simpler setup, less control
- Requires: `POLYMARKET_API_KEY`

```python
executor = PolymarketExecutor(
    private_key="<polygon_wallet_key>",
    api_key="<polymarket_api_key>",   # Optional if using CLOB
    mode="clob",                       # or "rest"
    dry_run=False,                     # True for testing
)

# Place order
order = executor.place_order(
    market_id="0x123...",
    side="BUY_YES",                    # or BUY_NO, SELL_YES, SELL_NO
    amount=85,                         # USDC amount
    price=0.35,                        # Limit price
    kelly_fraction=0.25,               # Position sizing
)

# Returns:
{
    "order_id": "0xabc...",
    "status": "submitted",
    "market_id": "0x123...",
    "side": "BUY_YES",
    "amount": 21.25,                   # $85 * 0.25 Kelly
    "price": 0.35,
    "tx_hash": "0xdef..."              # Blockchain tx
}
```

---

## Setup

### 1. Install Dependencies

```bash
# Core (optional for prediction-only)
pip install requests

# For blockchain execution (CLOB mode)
pip install py-clob-client

# For live data fetching (if not using public API)
# (none — Polymarket REST API is free and requires no installation)
```

### 2. Environment Variables

Create/update `.env`:

```bash
# Required for CLOB mode (blockchain trading)
POLYMARKET_PRIVATE_KEY=abc123...    # Polygon wallet private key (no 0x prefix)

# Optional for REST mode (API-based trading)
POLYMARKET_API_KEY=xyz789...        # Polymarket API key

# Configuration
POLYMARKET_MODE=clob                # "clob" or "rest"
POLYMARKET_DRY_RUN=True             # Set to False for live trading
```

### 3. Fund Wallet (for CLOB mode)

1. Get your Polygon wallet address:
   ```bash
   python -c "from eth_keys import keys; print(keys.PrivateKey(bytes.fromhex('<your_key>')).public_key.to_checksum_address())"
   ```
2. Bridge USDC to Polygon (use [Polygon Bridge](https://wallet.polygon.technology/bridge))
3. Verify balance in executor: `executor.get_balance()`

---

## API Endpoints

### Market Discovery

**`GET /api/polymarket/catalog`**
```bash
curl "http://localhost:5000/api/polymarket/catalog?limit=50&offset=0"
```

**`GET /api/polymarket/search?q=<query>`**
```bash
curl "http://localhost:5000/api/polymarket/search?q=Fed%20rate%20cut&limit=20"
```

**`GET /api/polymarket/category/<category>`**
```bash
curl "http://localhost:5000/api/polymarket/category/economics?limit=50"
```

**`GET /api/polymarket/markets/<market_id>`**
```bash
curl "http://localhost:5000/api/polymarket/markets/0x123abc..."
```

### Predictions

**`POST /api/polymarket/predict`**
```bash
curl -X POST http://localhost:5000/api/polymarket/predict \
  -H "Content-Type: application/json" \
  -d '{
    "market_id": "0x123...",
    "question": "Will the Fed cut rates in June?",
    "yes_price": 0.35,
    "no_price": 0.65,
    "volume": 52000,
    "category": "economics",
    "days_to_close": 95
  }'
```

**`POST /api/polymarket/predict/batch`**
```bash
curl -X POST http://localhost:5000/api/polymarket/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "markets": [
      {"id": "0x123", "question": "...", "yes_price": 0.35, ...},
      {"id": "0x456", "question": "...", "yes_price": 0.62, ...}
    ]
  }'
```

**`GET /api/polymarket/scan-opportunities`**
```bash
# Automatically scan all high-liquidity markets and return top trading opportunities
curl "http://localhost:5000/api/polymarket/scan-opportunities"
```

### Trading

**`POST /api/polymarket/trade`** (requires `POLYMARKET_PRIVATE_KEY`)
```bash
curl -X POST http://localhost:5000/api/polymarket/trade \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{
    "market_id": "0x123...",
    "side": "BUY_YES",
    "amount": 85,
    "price": 0.35,
    "kelly_fraction": 0.25
  }'
```

**`GET /api/polymarket/orders`** (get active orders)
```bash
curl "http://localhost:5000/api/polymarket/orders"
```

**`GET /api/polymarket/positions`** (get open positions)
```bash
curl "http://localhost:5000/api/polymarket/positions"
```

**`GET /api/polymarket/status`** (health check)
```bash
curl "http://localhost:5000/api/polymarket/status"
```

---

## Trading Pipeline

### Full Workflow

```python
from app.services.polymarket_data_fetcher import PolymarketDataFetcher
from app.ml.polymarket_predictor import PolymarketPredictor
from app.services.polymarket_executor import PolymarketExecutor
import os

# 1. FETCH markets with edge opportunities
data_fetcher = PolymarketDataFetcher()
predictor = PolymarketPredictor()

# Scan all high-liquidity markets
opportunities = data_fetcher.find_mispriced_markets(predictor)

# 2. PREDICT on each market
for opp in opportunities[:5]:  # Top 5 edges
    market_id = opp["market_id"]
    question = opp["question"]
    yes_price = opp["market_yes_price"]

    prediction = predictor.predict(
        market_id=market_id,
        question=question,
        yes_price=yes_price,
        volume=opp["volume"],
    )

    print(f"Edge: {prediction['edge_signal']}")
    print(f"Order size: ${prediction['suggested_order_size']}")

    # 3. EXECUTE if edge is significant
    if prediction["edge"] > 0.05:  # >5% edge
        executor = PolymarketExecutor(
            private_key=os.environ.get("POLYMARKET_PRIVATE_KEY"),
            dry_run=False,  # ⚠️ For live trading, set to False
        )

        order = executor.place_order(
            market_id=market_id,
            side=prediction["edge_signal"].split()[0] + "_" +
                 ("YES" if "BUY YES" in prediction["edge_signal"] else "NO"),
            amount=prediction["suggested_order_size"],
            price=yes_price,
            kelly_fraction=prediction["kelly_fraction"],
        )

        print(f"Order placed: {order['order_id']}")
        print(f"Status: {order['status']}")
```

---

## Features & Capabilities

### Data Layer
- ✅ Fetch all active Polymarket markets
- ✅ Search markets by keyword
- ✅ Filter by category (economics, politics, sports, crypto, etc.)
- ✅ Get current bid/ask prices
- ✅ Track order book and recent trades
- ✅ Liquidity scoring (0-100)
- ✅ Auto-scan for highest-edge markets

### Prediction Layer
- ✅ Calibrated YES/NO probabilities
- ✅ Base rate adjustments (Fed, CPI, elections, sports)
- ✅ Economic context integration (FRED indicators)
- ✅ Liquidity-adjusted confidence
- ✅ Kelly criterion position sizing (fractional)
- ✅ Edge signal ("BUY YES", "BUY NO", "NEUTRAL")
- ✅ Batch prediction (1000s of markets at once)

### Execution Layer
- ✅ CLOB mode: Sign & submit orders directly to blockchain
- ✅ REST mode: Use Polymarket API for order placement
- ✅ Dry-run mode: Simulate trades without real funds
- ✅ Track order status and fills
- ✅ Position management (open, close)
- ✅ Balance inquiry
- ✅ Health checks

### Risk Management
- ✅ Kelly fractional sizing (typically 0.25 Kelly)
- ✅ Minimum edge threshold (4% to trade)
- ✅ Liquidity-based sizing adjustments
- ✅ Confidence-weighted position sizing
- ✅ Dry-run mode for testing

---

## Best Practices

### 1. **Start with Dry-Run**
```python
executor = PolymarketExecutor(
    private_key=os.environ.get("POLYMARKET_PRIVATE_KEY"),
    dry_run=True,  # ← Always start here
)
```

### 2. **Use Quarter Kelly**
Kelly fraction of 0.25 is recommended for live trading.
- Full Kelly = too risky for model uncertainty
- Quarter Kelly = reasonable risk/reward

### 3. **Filter by Liquidity**
Avoid thin markets — they have wider spreads and harder execution.
```python
# Only trade markets with >$5k 24h volume and 10+ traders
markets = data_fetcher.filter_high_liquidity_markets(
    min_volume=5000,
    min_traders=10
)
```

### 4. **Check Edge Threshold**
```python
if abs(prediction["edge"]) > 0.05:  # >5% edge
    # Trade
else:
    # Skip — not enough edge to overcome fees
```

### 5. **Monitor Economic Context**
For economic questions (Fed, CPI, unemployment), provide economic indicators:
```python
from app.ml.kalshi_predictor import KalshiPredictor

kalshi = KalshiPredictor()
econ_context = kalshi.get_economic_context()

prediction = predictor.predict(
    ...,
    economic_context=econ_context,  # Fed rates, CPI, etc.
)
```

---

## Troubleshooting

### "py-clob-client not installed"
- The system defaults to REST mode if CLOB is unavailable
- For CLOB mode, install: `pip install py-clob-client`
- For REST-only, you're fine — no installation needed

### "Order placement failed: Unauthorized"
- Check `POLYMARKET_API_KEY` is set correctly (for REST mode)
- Verify wallet has USDC balance on Polygon (for CLOB mode)

### "Market not found"
- Market might be closed or archived
- Refresh the market list: `data_fetcher.get_active_markets()`

### "Spread too wide"
- Market has low liquidity
- Filter: `min_volume=5000, min_traders=10`

---

## Comparison: Kalshi vs Polymarket

| Aspect | Kalshi | Polymarket |
|--------|--------|-----------|
| **Blockchain** | None (centralized) | Polygon |
| **Authentication** | API key + password | Private key (CLOB) / API key (REST) |
| **Order Book** | Order matching | CLOB |
| **Currency** | USD (fiat) | USDC (stablecoin) |
| **Liquidity** | Varies | Generally high |
| **Fees** | Maker/taker | Minimal on-chain |
| **Prediction Model** | KalshiPredictor | PolymarketPredictor (extends Kalshi) |
| **Execution** | REST API only | CLOB (direct) or REST API |

---

## API Reference

See `/api/polymarket.py` for all endpoint signatures, or run `/api/polymarket/status` for health check.

---

## Next Steps

1. **Test predictions** without trading:
   ```bash
   curl http://localhost:5000/api/polymarket/scan-opportunities
   ```

2. **Set up wallet** (for blockchain mode):
   - Generate key via MetaMask or similar
   - Bridge USDC to Polygon
   - Set `POLYMARKET_PRIVATE_KEY` env var

3. **Run dry-run trades** to verify execution pipeline:
   ```python
   executor = PolymarketExecutor(dry_run=True)
   order = executor.place_order(...)
   ```

4. **Go live** (carefully):
   - Start with small position sizes
   - Monitor P&L closely
   - Gradually increase size as confidence grows

---

**Questions?** See the implementation in:
- `app/services/polymarket_data_fetcher.py`
- `app/ml/polymarket_predictor.py`
- `app/services/polymarket_executor.py`
- `app/api/polymarket.py`
