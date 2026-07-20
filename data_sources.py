"""
data_sources.py
================
Two families of data source:

1. SYNTHETIC (used in this repo's demo/backtest): generates a panel of
   log-prices from a VECM data-generating process with a KNOWN alpha, beta,
   and rank. This is the standard way to unit-test a cointegration pipeline
   -- you never trust that your Johansen/VECM code is correct just because
   it runs without error; you first check it can recover parameters you
   planted yourself. Only after that should you point it at real markets.

2. LIVE (yfinance / Binance / Alpaca): thin wrappers you can run on your own
   machine. They are NOT executed in this sandbox because the sandbox's
   network egress is restricted to package registries (no api access to
   Yahoo/Binance/Alpaca from here) -- but the functions are complete and
   ready to use locally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ====================================================================== #
# 1. SYNTHETIC VECM DATA GENERATOR
# ====================================================================== #
def generate_vecm_panel(
    n_assets: int = 6,
    n_obs: int = 1500,
    rank: int = 2,
    speed_range: tuple[float, float] = (-0.08, -0.02),
    noise_std: float = 0.006,
    trend_vol: float = 0.010,
    seed: int = 7,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Simulates:
        Δy_t = alpha @ beta.T @ y_{t-1} + eps_t

    which integrates into log-price levels y_t that are cointegrated with
    exactly `rank` independent stationary combinations (and n_assets - rank
    common stochastic trends).

    Returns
    -------
    prices : DataFrame of log-prices, columns ASSET_0..ASSET_{k-1}, business-day index
    true_beta  : (k, rank) ground-truth cointegrating vectors
    true_alpha : (k, rank) ground-truth speed-of-adjustment
    """
    rng = np.random.default_rng(seed)
    k = n_assets

    # --- ground-truth cointegrating vectors (beta) ---
    # Each vector says "asset_0 minus some weighted basket of the others is
    # stationary". We keep weights modest and positive-ish so spreads are
    # interpretable as long/short baskets.
    true_beta = np.zeros((k, rank))
    for r in range(rank):
        vec = rng.uniform(0.3, 1.2, size=k)
        vec[r] = 1.0
        # zero out weight on assets already "used up" by earlier vectors
        # to keep the vectors closer to independent
        if r > 0:
            vec[:r] = 0.0
        true_beta[:, r] = vec

    # --- ground-truth speed of adjustment (alpha) ---
    # Negative = error-correcting (spread shrinks back toward 0), which is
    # what a mean-reverting relationship requires. Magnitude controls how
    # many periods it takes to correct a deviation.
    true_alpha = np.zeros((k, rank))
    for r in range(rank):
        loaded_assets = np.where(true_beta[:, r] != 0)[0]
        speeds = rng.uniform(*speed_range, size=len(loaded_assets))
        true_alpha[loaded_assets, r] = speeds

    # --- simulate the VECM path ---
    # NOTE: the pure recursion  y_t = y_{t-1} + alpha @ (beta' y_{t-1}) + eps_t
    # is ALREADY exactly what we need: its companion matrix
    # Phi = I + alpha @ beta' has exactly r eigenvalues strictly inside the
    # unit circle (the r stationary error-correction directions) and
    # k - r eigenvalues exactly equal to 1 (the common stochastic trends).
    # No separate trend needs to be bolted on -- doing so (an earlier version
    # of this function did) risks injecting a component that isn't orthogonal
    # to beta and silently breaks the planted cointegrating relationship.
    y = np.zeros((n_obs, k))
    y[0] = rng.normal(scale=1.0, size=k)

    for t in range(1, n_obs):
        ec_term = true_alpha @ (true_beta.T @ y[t - 1])
        eps = rng.normal(0, noise_std, size=k)
        y[t] = y[t - 1] + ec_term + eps

    dates = pd.bdate_range("2021-01-04", periods=n_obs)
    cols = [f"ASSET_{i}" for i in range(k)]
    prices = pd.DataFrame(y, index=dates, columns=cols)
    return prices, true_beta, true_alpha


def synthetic_order_book_ticks(
    log_prices: pd.DataFrame, ticks_per_bar: int = 20, seed: int = 11
) -> pd.DataFrame:
    """
    Expands daily bars into a higher-frequency tick stream (for feeding the
    Redis pub/sub pipeline in the live-demo sense) by Brownian-bridging
    between consecutive closes and attaching a synthetic bid/ask spread.
    Used only to make the distributed pipeline demo realistic; the actual
    statistical backtest runs on the bar data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    assets = log_prices.columns
    for asset in assets:
        s = log_prices[asset].values
        for i in range(len(s) - 1):
            start, end = s[i], s[i + 1]
            bridge = np.linspace(start, end, ticks_per_bar + 1)[1:]
            noise = rng.normal(0, abs(end - start) * 0.15 + 1e-6, size=ticks_per_bar)
            mid = bridge + noise
            spread_bps = rng.uniform(1, 8, size=ticks_per_bar)  # 1-8 bps
            for j in range(ticks_per_bar):
                rows.append(
                    {
                        "bar_idx": i,
                        "tick_idx": j,
                        "asset": asset,
                        "mid_price": np.exp(mid[j]),
                        "bid": np.exp(mid[j]) * (1 - spread_bps[j] / 2 / 1e4),
                        "ask": np.exp(mid[j]) * (1 + spread_bps[j] / 2 / 1e4),
                    }
                )
    return pd.DataFrame(rows)


# ====================================================================== #
# 2. LIVE DATA SOURCES (ready to run locally; not called in this sandbox)
# ====================================================================== #
def fetch_yahoo_daily(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily adjusted close panel via yfinance. Run this on your own machine."""
    import yfinance as yf  # local import: optional dependency

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return np.log(raw.dropna())


def fetch_binance_klines(symbols: list[str], interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    """
    Minute-level crypto klines via Binance public REST API (no key required
    for market data). Run this on your own machine.
    """
    import requests

    frames = {}
    for sym in symbols:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": sym, "interval": interval, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
        closes = [float(row[4]) for row in raw]
        times = pd.to_datetime([row[0] for row in raw], unit="ms")
        frames[sym] = pd.Series(closes, index=times)
    df = pd.DataFrame(frames).dropna()
    return np.log(df)


def fetch_polygon_minute_bars(tickers: list[str], api_key: str, start: str, end: str) -> pd.DataFrame:
    """Minute-level ETF/sector bars via Polygon.io. Requires POLYGON_API_KEY."""
    import requests

    frames = {}
    for t in tickers:
        url = f"https://api.polygon.io/v2/aggs/ticker/{t}/range/1/minute/{start}/{end}"
        resp = requests.get(url, params={"apiKey": api_key, "adjusted": "true", "limit": 50000}, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        closes = [r["c"] for r in results]
        times = pd.to_datetime([r["t"] for r in results], unit="ms")
        frames[t] = pd.Series(closes, index=times)
    df = pd.DataFrame(frames).dropna()
    return np.log(df)
