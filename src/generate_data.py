"""
Generate a realistic synthetic crypto trades dataset.

- 211,000+ rows
- 32 accounts (multi-schema: retail / prop / market_maker tiers)
- 8 symbols across 3 venues
- Geometric Brownian Motion price paths with regime shifts
  (so the downstream regime/volatility analysis is meaningful, not noise)

Output: data/raw/crypto_trades.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_ROWS_TARGET = 211_000
N_ACCOUNTS = 32
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
           "AVAX-USD", "MATIC-USD", "DOT-USD", "LINK-USD"]
VENUES = ["coinbase", "binance", "kraken"]
SIDES = ["BUY", "SELL"]

# Three account tiers — gives us the "multi-schema across 32 accounts" angle.
# Each tier has a different fee structure and trade-size distribution.
TIERS = ["retail", "prop", "market_maker"]
TIER_WEIGHTS = [0.55, 0.30, 0.15]


def build_account_book() -> pd.DataFrame:
    tiers = RNG.choice(TIERS, size=N_ACCOUNTS, p=TIER_WEIGHTS)
    accounts = pd.DataFrame({
        "account_id": [f"ACC{1000+i}" for i in range(N_ACCOUNTS)],
        "tier": tiers,
        "fee_bps": np.where(tiers == "retail", 25.0,
                    np.where(tiers == "prop", 8.0, 2.0)),
        "base_trade_notional": np.where(tiers == "retail", 500.0,
                               np.where(tiers == "prop", 25_000.0, 250_000.0)),
    })
    return accounts


def build_price_paths(start: datetime, minutes: int) -> dict[str, np.ndarray]:
    """One GBM path per symbol, with 3 volatility regimes injected.

    Regime structure (shared across symbols so cross-asset analysis is honest):
      first third -> low vol, middle -> high vol, last -> medium vol
    """
    paths: dict[str, np.ndarray] = {}
    # base prices roughly Jan-2024-ish, doesn't have to be exact
    base_px = {"BTC-USD": 42_000, "ETH-USD": 2_300, "SOL-USD": 95,
               "ADA-USD": 0.55, "AVAX-USD": 38, "MATIC-USD": 0.85,
               "DOT-USD": 7.5, "LINK-USD": 15}

    t1, t2 = minutes // 3, 2 * minutes // 3
    # per-minute sigma per regime
    sigma = np.empty(minutes)
    sigma[:t1] = 0.0008
    sigma[t1:t2] = 0.0035
    sigma[t2:] = 0.0018
    mu = 0.00002  # slight drift

    for sym, p0 in base_px.items():
        shocks = RNG.normal(loc=mu, scale=sigma, size=minutes)
        # symbol-specific extra noise scale
        shocks *= RNG.uniform(0.8, 1.4)
        log_path = np.log(p0) + np.cumsum(shocks)
        paths[sym] = np.exp(log_path)
    return paths


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "crypto_trades.csv")

    accounts = build_account_book()
    print(f"Built {len(accounts)} accounts across tiers: "
          f"{accounts['tier'].value_counts().to_dict()}")

    # Trade timeline: ~60 days at variable per-minute intensity
    start = datetime(2024, 1, 1)
    minutes = 60 * 24 * 60  # 60 days of minutes = 86,400
    prices = build_price_paths(start, minutes)

    # Generate N_ROWS_TARGET trades, sampling minute offsets with
    # higher density in the high-vol middle regime (realistic).
    weights = np.ones(minutes)
    weights[minutes // 3: 2 * minutes // 3] *= 3.0  # busier in chaos
    weights /= weights.sum()
    minute_idx = RNG.choice(minutes, size=N_ROWS_TARGET, p=weights)
    # within-minute jitter in seconds
    second_jitter = RNG.integers(0, 60, size=N_ROWS_TARGET)

    # sample symbols and accounts
    sym_idx = RNG.integers(0, len(SYMBOLS), size=N_ROWS_TARGET)
    symbols = np.array(SYMBOLS)[sym_idx]
    acct_idx = RNG.integers(0, N_ACCOUNTS, size=N_ROWS_TARGET)
    account_ids = accounts["account_id"].values[acct_idx]
    tiers = accounts["tier"].values[acct_idx]
    fees_bps = accounts["fee_bps"].values[acct_idx]
    base_notional = accounts["base_trade_notional"].values[acct_idx]

    # mid prices at sampled minutes per symbol
    mid = np.array([prices[s][m] for s, m in zip(symbols, minute_idx)])
    # spread by venue (bps)
    venue_idx = RNG.integers(0, len(VENUES), size=N_ROWS_TARGET)
    venues = np.array(VENUES)[venue_idx]
    spread_bps = np.where(venues == "coinbase", 4.0,
                  np.where(venues == "binance", 2.0, 6.0))

    sides = RNG.choice(SIDES, size=N_ROWS_TARGET)
    side_sign = np.where(sides == "BUY", 1.0, -1.0)
    fill_price = mid * (1 + side_sign * spread_bps / 2 / 10_000.0)

    # quantity ~ Lognormal around base_notional / price
    notional_mult = RNG.lognormal(mean=0.0, sigma=0.6, size=N_ROWS_TARGET)
    quantity = (base_notional * notional_mult) / fill_price
    # round qty sensibly per symbol
    quantity = np.where(mid > 1000, np.round(quantity, 4),
                np.where(mid > 10, np.round(quantity, 2), np.round(quantity, 1)))

    notional = fill_price * quantity
    fee_amount = notional * fees_bps / 10_000.0

    # PnL placeholder: realized PnL vs end-of-day VWAP of same symbol.
    # We'll just use price drift from this trade vs +30min mid as a proxy.
    horizon = 30
    fwd_minute = np.clip(minute_idx + horizon, 0, minutes - 1)
    fwd_px = np.array([prices[s][m] for s, m in zip(symbols, fwd_minute)])
    # signed exposure pnl
    realized_pnl = side_sign * (fwd_px - fill_price) * quantity - fee_amount

    ts = np.array([start + timedelta(minutes=int(m), seconds=int(s))
                   for m, s in zip(minute_idx, second_jitter)])

    df = pd.DataFrame({
        "trade_id":      [f"T{1_000_000+i}" for i in range(N_ROWS_TARGET)],
        "ts":            ts,
        "account_id":    account_ids,
        "tier":          tiers,
        "symbol":        symbols,
        "venue":         venues,
        "side":          sides,
        "quantity":      quantity.astype(np.float64),
        "fill_price":    np.round(fill_price, 6),
        "mid_price":     np.round(mid, 6),
        "notional":      np.round(notional, 4),
        "fee_bps":       fees_bps,
        "fee_amount":    np.round(fee_amount, 6),
        "realized_pnl":  np.round(realized_pnl, 6),
    })

    df = df.sort_values("ts").reset_index(drop=True)
    df["trade_date"] = df["ts"].dt.date.astype(str)  # for Hive-style partitioning

    df.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Wrote {len(df):,} rows -> {out_path}  ({size_mb:.1f} MB)")
    print(df.head())
    print("\nPer-day row counts (sample):")
    print(df["trade_date"].value_counts().sort_index().head(10))


if __name__ == "__main__":
    main()
