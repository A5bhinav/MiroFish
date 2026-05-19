"""
Daily market snapshot collector.

Run once per day (via cron or manually) to build a growing training dataset
of market prices paired with eventual resolutions.

Usage:
    uv run python scripts/collect_snapshots.py [--resolve-only]

  --resolve-only   Skip new snapshots; only check existing open markets for resolution.

Schedule example (cron, run at 9am UTC daily):
    0 9 * * * cd /path/to/MiroFish/backend && uv run python scripts/collect_snapshots.py

After ~2 months of daily collection you'll have several hundred resolved
market pairs suitable for re-training the Kalshi calibrator.
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("collect_snapshots")


def main():
    parser = argparse.ArgumentParser(description="Collect / resolve market snapshots")
    parser.add_argument("--resolve-only", action="store_true",
                        help="Only check for resolutions, don't snapshot new markets")
    parser.add_argument("--max", type=int, default=200,
                        help="Max markets to snapshot per platform (default 200)")
    args = parser.parse_args()

    # Lazy imports (avoid Flask startup)
    from app.ml.data_pipeline import collect_market_snapshots, resolve_market_snapshots

    kalshi_fetcher = None
    polymarket_fetcher = None

    # --- Kalshi (optional — requires credentials) ---
    try:
        from app.services.kalshi_data_fetcher import KalshiDataFetcher
        from app.config import Config
        if Config.KALSHI_API_KEY and Config.KALSHI_PRIVATE_KEY:
            kalshi_fetcher = KalshiDataFetcher()
            logger.info("Kalshi fetcher initialized")
        else:
            logger.info("Kalshi credentials not set — skipping Kalshi markets")
    except Exception as e:
        logger.warning(f"Kalshi fetcher unavailable: {e}")

    # --- Polymarket (no key required) ---
    try:
        from app.services.polymarket_data_fetcher import PolymarketDataFetcher
        polymarket_fetcher = PolymarketDataFetcher()
        logger.info("Polymarket fetcher initialized")
    except Exception as e:
        logger.warning(f"Polymarket fetcher unavailable: {e}")

    # --- Resolve existing open snapshots ---
    resolved = resolve_market_snapshots(
        kalshi_fetcher=kalshi_fetcher,
        polymarket_fetcher=polymarket_fetcher,
    )
    logger.info(f"Resolved {resolved} markets this run")

    # --- Snapshot new open markets ---
    if not args.resolve_only:
        saved = collect_market_snapshots(
            kalshi_fetcher=kalshi_fetcher,
            polymarket_fetcher=polymarket_fetcher,
            max_markets=args.max,
        )
        logger.info(f"Snapshotted {saved} new markets")

    logger.info("Done.")


if __name__ == "__main__":
    main()
